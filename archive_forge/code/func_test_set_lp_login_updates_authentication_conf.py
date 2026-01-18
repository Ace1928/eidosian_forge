from ... import config
from ...tests import TestCaseInTempDir, TestCaseWithMemoryTransport
from . import account
def test_set_lp_login_updates_authentication_conf(self):
    self.assertIs(None, account._get_auth_user())
    account.set_lp_login('foo')
    self.assertEqual('foo', account._get_auth_user())