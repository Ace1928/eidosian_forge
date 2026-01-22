from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
def assertPreAndPostHooksWereInvoked(self, branch, smart_enabled):
    """assert that both pre and post hooks were called

        :param smart_enabled: The method invoked is one that should be
            smart server ready.
        """
    if smart_enabled and isinstance(branch, remote.RemoteBranch):
        length = 2
    else:
        length = 1
    self.assertEqual(length, len(self.pre_hook_calls))
    self.assertEqual(length, len(self.post_hook_calls))