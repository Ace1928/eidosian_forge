from ... import config
from ...tests import TestCaseInTempDir, TestCaseWithMemoryTransport
from . import account
def test_set_lp_login(self):
    my_config = config.MemoryStack(b'')
    self.assertEqual(None, my_config.get('launchpad_username'))
    account.set_lp_login('test-user', my_config)
    self.assertEqual('test-user', my_config.get('launchpad_username'))