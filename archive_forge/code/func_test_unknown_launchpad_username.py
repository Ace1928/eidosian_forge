from ... import config
from ...tests import TestCaseInTempDir, TestCaseWithMemoryTransport
from . import account
def test_unknown_launchpad_username(self):
    error = account.UnknownLaunchpadUsername(user='test-user')
    self.assertEqualDiff('The user name test-user is not registered on Launchpad.', str(error))