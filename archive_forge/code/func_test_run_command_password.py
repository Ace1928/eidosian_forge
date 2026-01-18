import socket
import threading
from io import StringIO
from unittest import skipIf
from dulwich.tests import TestCase
def test_run_command_password(self):
    vendor = ParamikoSSHVendor(allow_agent=False, look_for_keys=False)
    vendor.run_command('127.0.0.1', 'test_run_command_password', username=USER, port=self.port, password=PASSWORD)
    self.assertIn(b'test_run_command_password', self.commands)