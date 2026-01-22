from __future__ import (absolute_import, division, print_function)
import fcntl
import hashlib
import io
import os
import pickle
import signal
import socket
import sys
import time
import traceback
import errno
import json
from contextlib import contextmanager
from ansible import constants as C
from ansible.cli.arguments import option_helpers as opt_help
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError, send_data, recv_data
from ansible.module_utils.service import fork_process
from ansible.parsing.ajson import AnsibleJSONEncoder, AnsibleJSONDecoder
from ansible.playbook.play_context import PlayContext
from ansible.plugins.loader import connection_loader, init_plugin_loader
from ansible.utils.path import unfrackpath, makedirs_safe
from ansible.utils.display import Display
from ansible.utils.jsonrpc import JsonRpcServer
class ConnectionProcess(object):
    """
    The connection process wraps around a Connection object that manages
    the connection to a remote device that persists over the playbook
    """

    def __init__(self, fd, play_context, socket_path, original_path, task_uuid=None, ansible_playbook_pid=None):
        self.play_context = play_context
        self.socket_path = socket_path
        self.original_path = original_path
        self._task_uuid = task_uuid
        self.fd = fd
        self.exception = None
        self.srv = JsonRpcServer()
        self.sock = None
        self.connection = None
        self._ansible_playbook_pid = ansible_playbook_pid

    def start(self, options):
        messages = list()
        result = {}
        try:
            messages.append(('vvvv', 'control socket path is %s' % self.socket_path))
            if self.play_context.private_key_file and self.play_context.private_key_file[0] not in '~/':
                self.play_context.private_key_file = os.path.join(self.original_path, self.play_context.private_key_file)
            self.connection = connection_loader.get(self.play_context.connection, self.play_context, '/dev/null', task_uuid=self._task_uuid, ansible_playbook_pid=self._ansible_playbook_pid)
            try:
                self.connection.set_options(direct=options)
            except ConnectionError as exc:
                messages.append(('debug', to_text(exc)))
                raise ConnectionError('Unable to decode JSON from response set_options. See the debug log for more information.')
            self.connection._socket_path = self.socket_path
            self.srv.register(self.connection)
            messages.extend([('vvvv', msg) for msg in sys.stdout.getvalue().splitlines()])
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.bind(self.socket_path)
            self.sock.listen(1)
            messages.append(('vvvv', 'local domain socket listeners started successfully'))
        except Exception as exc:
            messages.extend(self.connection.pop_messages())
            result['error'] = to_text(exc)
            result['exception'] = traceback.format_exc()
        finally:
            result['messages'] = messages
            self.fd.write(json.dumps(result, cls=AnsibleJSONEncoder))
            self.fd.close()

    def run(self):
        try:
            log_messages = self.connection.get_option('persistent_log_messages')
            while not self.connection._conn_closed:
                signal.signal(signal.SIGALRM, self.connect_timeout)
                signal.signal(signal.SIGTERM, self.handler)
                signal.alarm(self.connection.get_option('persistent_connect_timeout'))
                self.exception = None
                s, addr = self.sock.accept()
                signal.alarm(0)
                signal.signal(signal.SIGALRM, self.command_timeout)
                while True:
                    data = recv_data(s)
                    if not data:
                        break
                    if log_messages:
                        display.display('jsonrpc request: %s' % data, log_only=True)
                    request = json.loads(to_text(data, errors='surrogate_or_strict'))
                    if request.get('method') == 'exec_command' and (not self.connection.connected):
                        self.connection._connect()
                    signal.alarm(self.connection.get_option('persistent_command_timeout'))
                    resp = self.srv.handle_request(data)
                    signal.alarm(0)
                    if log_messages:
                        display.display('jsonrpc response: %s' % resp, log_only=True)
                    send_data(s, to_bytes(resp))
                s.close()
        except Exception as e:
            if hasattr(e, 'errno'):
                if e.errno != errno.EINTR:
                    self.exception = traceback.format_exc()
            else:
                self.exception = traceback.format_exc()
        finally:
            time.sleep(0.1)
            self.shutdown()

    def connect_timeout(self, signum, frame):
        msg = 'persistent connection idle timeout triggered, timeout value is %s secs.\nSee the timeout setting options in the Network Debug and Troubleshooting Guide.' % self.connection.get_option('persistent_connect_timeout')
        display.display(msg, log_only=True)
        raise Exception(msg)

    def command_timeout(self, signum, frame):
        msg = 'command timeout triggered, timeout value is %s secs.\nSee the timeout setting options in the Network Debug and Troubleshooting Guide.' % self.connection.get_option('persistent_command_timeout')
        display.display(msg, log_only=True)
        raise Exception(msg)

    def handler(self, signum, frame):
        msg = 'signal handler called with signal %s.' % signum
        display.display(msg, log_only=True)
        raise Exception(msg)

    def shutdown(self):
        """ Shuts down the local domain socket
        """
        lock_path = unfrackpath('%s/.ansible_pc_lock_%s' % os.path.split(self.socket_path))
        if os.path.exists(self.socket_path):
            try:
                if self.sock:
                    self.sock.close()
                if self.connection:
                    self.connection.close()
                    if self.connection.get_option('persistent_log_messages'):
                        for _level, message in self.connection.pop_messages():
                            display.display(message, log_only=True)
            except Exception:
                pass
            finally:
                if os.path.exists(self.socket_path):
                    os.remove(self.socket_path)
                    setattr(self.connection, '_socket_path', None)
                    setattr(self.connection, '_connected', False)
        if os.path.exists(lock_path):
            os.remove(lock_path)
        display.display('shutdown complete', log_only=True)