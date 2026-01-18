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
def test_run_with_later_install_cmd(self):
    code, out, err = self.execute(['later_install_cmd'])
    self.assertEqual(cmd.RC_NOEXECFOUND, code)
    shutil.copy('/bin/echo', self.later_cmd)
    code, out, err = self.execute(['later_install_cmd'])
    self.assertEqual(0, code)