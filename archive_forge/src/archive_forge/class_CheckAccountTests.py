from ... import config
from ...tests import TestCaseInTempDir, TestCaseWithMemoryTransport
from . import account
class CheckAccountTests(TestCaseWithMemoryTransport):

    def test_check_lp_login_valid_user(self):
        transport = self.get_transport()
        transport.mkdir('~test-user')
        transport.put_bytes('~test-user/+sshkeys', b'some keys here')
        account.check_lp_login('test-user', transport)

    def test_check_lp_login_no_user(self):
        transport = self.get_transport()
        self.assertRaises(account.UnknownLaunchpadUsername, account.check_lp_login, 'test-user', transport)

    def test_check_lp_login_no_ssh_keys(self):
        transport = self.get_transport()
        transport.mkdir('~test-user')
        transport.put_bytes('~test-user/+sshkeys', b'')
        self.assertRaises(account.NoRegisteredSSHKeys, account.check_lp_login, 'test-user', transport)