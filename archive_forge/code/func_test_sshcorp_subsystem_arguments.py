from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
def test_sshcorp_subsystem_arguments(self):
    vendor = SSHCorpSubprocessVendor()
    self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, subsystem='sftp'), ['ssh', '-x', '-p', '100', '-l', 'user', '-s', 'sftp', 'host'])