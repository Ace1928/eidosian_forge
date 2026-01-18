import socket
import threading
from io import StringIO
from unittest import skipIf
from dulwich.tests import TestCase
def test_run_command_with_privkey(self):
    key = paramiko.RSAKey.from_private_key(StringIO(CLIENT_KEY))
    vendor = ParamikoSSHVendor(allow_agent=False, look_for_keys=False)
    vendor.run_command('127.0.0.1', 'test_run_command_with_privkey', username=USER, port=self.port, pkey=key)
    self.assertIn(b'test_run_command_with_privkey', self.commands)