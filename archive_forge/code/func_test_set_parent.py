from breezy.branch import UnstackableBranchFormat
from breezy.errors import IncompatibleFormat
from breezy.revision import NULL_REVISION
from breezy.tests import TestCaseWithTransport
def test_set_parent(self):
    """Test that setting the parent works."""
    branch = self.make_branch()
    branch.set_parent('foobar')