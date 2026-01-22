import linecache
import os
from _pydev_bundle.pydev_imports import _queue
from _pydev_bundle._pydev_saved_modules import time
from _pydev_bundle._pydev_saved_modules import threading
from _pydev_bundle._pydev_saved_modules import socket as socket_module
from _pydevd_bundle.pydevd_constants import (DebugInfoHolder, IS_WINDOWS, IS_JYTHON, IS_WASM,
from _pydev_bundle.pydev_override import overrides
import weakref
from _pydev_bundle._pydev_completer import extract_token_and_qualifier
from _pydevd_bundle._debug_adapter.pydevd_schema import VariablesResponseBody, \
from _pydevd_bundle._debug_adapter import pydevd_base_schema, pydevd_schema
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_xml import ExceptionOnEvaluate
from _pydevd_bundle.pydevd_constants import ForkSafeLock, NULL
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id, resume_threads
from _pydevd_bundle.pydevd_dont_trace_files import PYDEV_FILE
import dis
import pydevd_file_utils
import itertools
from urllib.parse import quote_plus, unquote_plus
import pydevconsole
from _pydevd_bundle import pydevd_vars, pydevd_io, pydevd_reload
from _pydevd_bundle import pydevd_bytecode_utils
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle import pydevd_vm_type
import sys
import traceback
from _pydevd_bundle.pydevd_utils import quote_smart as quote, compare_object_attrs_key, \
from _pydev_bundle import pydev_log, fsnotify
from _pydev_bundle.pydev_log import exception as pydev_log_exception
from _pydev_bundle import _pydev_completer
from pydevd_tracing import get_exception_traceback_str
from _pydevd_bundle import pydevd_console
from _pydev_bundle.pydev_monkey import disable_trace_thread_modules, enable_trace_thread_modules
from io import StringIO
from _pydevd_bundle.pydevd_comm_constants import *  # @UnusedWildImport
class ReaderThread(PyDBDaemonThread):
    """ reader thread reads and dispatches commands in an infinite loop """

    def __init__(self, sock, py_db, PyDevJsonCommandProcessor, process_net_command, terminate_on_socket_close=True):
        assert sock is not None
        PyDBDaemonThread.__init__(self, py_db)
        self.__terminate_on_socket_close = terminate_on_socket_close
        self.sock = sock
        self._buffer = b''
        self.name = 'pydevd.Reader'
        self.process_net_command = process_net_command
        self.process_net_command_json = PyDevJsonCommandProcessor(self._from_json).process_net_command_json

    def _from_json(self, json_msg, update_ids_from_dap=False):
        return pydevd_base_schema.from_json(json_msg, update_ids_from_dap, on_dict_loaded=self._on_dict_loaded)

    def _on_dict_loaded(self, dct):
        for listener in self.py_db.dap_messages_listeners:
            listener.after_receive(dct)

    @overrides(PyDBDaemonThread.do_kill_pydev_thread)
    def do_kill_pydev_thread(self):
        PyDBDaemonThread.do_kill_pydev_thread(self)

    def _read(self, size):
        while True:
            buffer_len = len(self._buffer)
            if buffer_len == size:
                ret = self._buffer
                self._buffer = b''
                return ret
            if buffer_len > size:
                ret = self._buffer[:size]
                self._buffer = self._buffer[size:]
                return ret
            try:
                r = self.sock.recv(max(size - buffer_len, 1024))
            except OSError:
                return b''
            if not r:
                return b''
            self._buffer += r

    def _read_line(self):
        while True:
            i = self._buffer.find(b'\n')
            if i != -1:
                i += 1
                ret = self._buffer[:i]
                self._buffer = self._buffer[i:]
                return ret
            else:
                try:
                    r = self.sock.recv(1024)
                except OSError:
                    return b''
                if not r:
                    return b''
                self._buffer += r

    @overrides(PyDBDaemonThread._on_run)
    def _on_run(self):
        try:
            content_len = -1
            while True:
                try:
                    notify_about_gevent_if_needed()
                    line = self._read_line()
                    if len(line) == 0:
                        pydev_log.debug('ReaderThread: empty contents received (len(line) == 0).')
                        self._terminate_on_socket_close()
                        return
                    if self._kill_received:
                        continue
                    if line.startswith(b'Content-Length:'):
                        content_len = int(line.strip().split(b':', 1)[1])
                        continue
                    if content_len != -1:
                        if line == b'\r\n':
                            json_contents = self._read(content_len)
                            content_len = -1
                            if len(json_contents) == 0:
                                pydev_log.debug('ReaderThread: empty contents received (len(json_contents) == 0).')
                                self._terminate_on_socket_close()
                                return
                            if self._kill_received:
                                continue
                            self.process_net_command_json(self.py_db, json_contents)
                        continue
                    elif line.endswith(b'\n\n'):
                        line = line[:-2]
                    elif line.endswith(b'\n'):
                        line = line[:-1]
                    elif line.endswith(b'\r'):
                        line = line[:-1]
                except:
                    if not self._kill_received:
                        pydev_log_exception()
                        self._terminate_on_socket_close()
                    return
                if hasattr(line, 'decode'):
                    line = line.decode('utf-8')
                if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 3:
                    pydev_log.debug('debugger: received >>%s<<\n', line)
                args = line.split('\t', 2)
                try:
                    cmd_id = int(args[0])
                    if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 3:
                        pydev_log.debug('Received command: %s %s\n', ID_TO_MEANING.get(str(cmd_id), '???'), line)
                    self.process_command(cmd_id, int(args[1]), args[2])
                except:
                    if sys is not None and pydev_log_exception is not None:
                        pydev_log_exception("Can't process net command: %s.", line)
        except:
            if not self._kill_received:
                if sys is not None and pydev_log_exception is not None:
                    pydev_log_exception()
            self._terminate_on_socket_close()
        finally:
            pydev_log.debug('ReaderThread: exit')

    def _terminate_on_socket_close(self):
        if self.__terminate_on_socket_close:
            self.py_db.dispose_and_kill_all_pydevd_threads()

    def process_command(self, cmd_id, seq, text):
        self.process_net_command(self.py_db, cmd_id, seq, text)