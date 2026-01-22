import contextlib
import io
import logging
import os
import pwd
import shutil
import signal
import sys
import threading
import time
from unittest import mock
import fixtures
import testtools
from testtools import content
from oslo_rootwrap import client
from oslo_rootwrap import cmd
from oslo_rootwrap import subprocess
from oslo_rootwrap.tests import run_daemon
@contextlib.contextmanager
def _test_daemon_cleanup(self):
    self.execute(['cat'])
    socket_path = self.client._manager._address
    yield
    process = self.client._process
    stop = threading.Event()

    def sleep_kill():
        stop.wait(1)
        if not stop.is_set():
            os.kill(process.pid, signal.SIGKILL)
    threading.Thread(target=sleep_kill).start()
    self.client._process.wait()
    stop.set()
    self.assertNotEqual(-signal.SIGKILL, process.returncode, "Server haven't stopped in one second")
    self.assertFalse(os.path.exists(socket_path), "Server didn't remove its temporary directory")