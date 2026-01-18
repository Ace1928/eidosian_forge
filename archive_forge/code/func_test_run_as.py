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
def test_run_as(self):
    if os.getuid() != 0:
        self.skip('Test requires root (for setuid)')
    code, out, err = self.execute(['id', '-u'])
    self.assertEqual('%s\n' % pwd.getpwnam('nobody').pw_uid, out)
    code, out, err = self.execute(['sh', '-c', 'id -u'])
    self.assertEqual('0\n', out)