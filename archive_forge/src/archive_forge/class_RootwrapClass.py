import functools
import logging
from multiprocessing import managers
import os
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time
from oslo_rootwrap import cmd
from oslo_rootwrap import jsonrpc
from oslo_rootwrap import subprocess
from oslo_rootwrap import wrapper
class RootwrapClass(object):

    def __init__(self, config, filters):
        self.config = config
        self.filters = filters
        self.reset_timer()
        self.prepare_timer(config)

    def run_one_command(self, userargs, stdin=None):
        self.reset_timer()
        try:
            obj = wrapper.start_subprocess(self.filters, userargs, exec_dirs=self.config.exec_dirs, log=self.config.use_syslog, close_fds=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except wrapper.FilterMatchNotExecutable:
            LOG.warning('Executable not found for: %s', ' '.join(userargs))
            return (cmd.RC_NOEXECFOUND, '', '')
        except wrapper.NoFilterMatched:
            LOG.warning('Unauthorized command: %s (no filter matched)', ' '.join(userargs))
            return (cmd.RC_UNAUTHORIZED, '', '')
        if stdin is not None:
            stdin = os.fsencode(stdin)
        out, err = obj.communicate(stdin)
        out = os.fsdecode(out)
        err = os.fsdecode(err)
        return (obj.returncode, out, err)

    @classmethod
    def reset_timer(cls):
        cls.last_called = time.time()

    @classmethod
    def cancel_timer(cls):
        try:
            cls.timeout.cancel()
        except RuntimeError:
            pass

    @classmethod
    def prepare_timer(cls, config=None):
        if config is not None:
            cls.daemon_timeout = config.daemon_timeout
        timeout = max(cls.last_called + cls.daemon_timeout - time.time(), 0) + 1
        if getattr(cls, 'timeout', None):
            return
        cls.timeout = threading.Timer(timeout, cls.handle_timeout)
        cls.timeout.start()

    @classmethod
    def handle_timeout(cls):
        if cls.last_called < time.time() - cls.daemon_timeout:
            cls.shutdown()
        cls.prepare_timer()

    @staticmethod
    def shutdown():
        os.kill(os.getpid(), signal.SIGINT)