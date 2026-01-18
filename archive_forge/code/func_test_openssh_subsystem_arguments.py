from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
def test_openssh_subsystem_arguments(self):
    vendor = OpenSSHSubprocessVendor()
    self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, subsystem='sftp'), ['ssh', '-oForwardX11=no', '-oForwardAgent=no', '-oClearAllForwardings=yes', '-oNoHostAuthenticationForLocalhost=yes', '-p', '100', '-l', 'user', '-s', '--', 'host', 'sftp'])