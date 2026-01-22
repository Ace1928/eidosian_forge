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
class Win32Server(BaseServer):

    def __init__(self, *args, **kwargs):
        super(Win32Server, self).__init__(*args, **kwargs)
        self._processutils = os_win_utilsfactory.get_processutils()

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
        if self.sock:
            self.sock.close()
            self.sock = None
        self.process = subprocess.Popen(cmd, env=self.exec_env)
        self.process_pid = self.process.pid
        try:
            self.job_handle = self._processutils.kill_process_on_job_close(self.process_pid)
        except Exception:
            self.process.kill()
            raise
        self.stop_kill = not expect_exit
        if self.pid_file:
            pf = open(self.pid_file, 'w')
            pf.write('%d\n' % self.process_pid)
            pf.close()
        rc = 0
        if expect_exit:
            self.process.communicate()
            rc = self.process.returncode
        return (rc, '', '')

    def stop(self):
        """
        Spin down the server.
        """
        if not self.process_pid:
            raise Exception('Server "%s" process not running.' % self.server_name)
        if self.stop_kill:
            self.process.terminate()
        return (0, '', '')