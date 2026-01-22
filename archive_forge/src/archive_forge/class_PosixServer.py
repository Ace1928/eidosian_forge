import abc
import atexit
import datetime
import errno
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from testtools import content as ttc
import textwrap
import time
from unittest import mock
import urllib.parse as urlparse
import uuid
import fixtures
import glance_store
from os_win import utilsfactory as os_win_utilsfactory
from oslo_config import cfg
from oslo_serialization import jsonutils
import testtools
import webob
from glance.common import config
from glance.common import utils
from glance.common import wsgi
from glance.db.sqlalchemy import api as db_api
from glance import tests as glance_tests
from glance.tests import utils as test_utils
import glance.async_
class PosixServer(BaseServer):

    def start(self, expect_exit=True, expected_exitcode=0, **kwargs):
        """
        Starts the server.

        Any kwargs passed to this method will override the configuration
        value in the conf file used in starting the servers.
        """
        self.write_conf(**kwargs)
        self.create_database()
        cmd = '%(server_module)s --config-file %(conf_file_name)s' % {'server_module': self.server_module, 'conf_file_name': self.conf_file_name}
        cmd = '%s -m %s' % (sys.executable, cmd)
        if self.exec_env:
            exec_env = self.exec_env.copy()
        else:
            exec_env = {}
        pass_fds = set()
        if self.sock:
            if not self.fork_socket:
                self.sock.close()
                self.sock = None
            else:
                fd = os.dup(self.sock.fileno())
                exec_env[utils.GLANCE_TEST_SOCKET_FD_STR] = str(fd)
                pass_fds.add(fd)
                self.sock.close()
        self.process_pid = test_utils.fork_exec(cmd, logfile=os.devnull, exec_env=exec_env, pass_fds=pass_fds)
        self.stop_kill = not expect_exit
        if self.pid_file:
            pf = open(self.pid_file, 'w')
            pf.write('%d\n' % self.process_pid)
            pf.close()
        if not expect_exit:
            rc = 0
            try:
                os.kill(self.process_pid, 0)
            except OSError:
                raise RuntimeError('The process did not start')
        else:
            rc = test_utils.wait_for_fork(self.process_pid, expected_exitcode=expected_exitcode, force=False)
        if self.sock:
            os.close(fd)
            self.sock = None
        return (rc, '', '')

    def stop(self):
        """
        Spin down the server.
        """
        if not self.process_pid:
            raise Exception('why is this being called? %s' % self.server_name)
        if self.stop_kill:
            os.kill(self.process_pid, signal.SIGTERM)
        rc = test_utils.wait_for_fork(self.process_pid, raise_error=False, force=self.stop_kill)
        return (rc, '', '')