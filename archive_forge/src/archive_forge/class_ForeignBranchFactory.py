from breezy.branch import UnstackableBranchFormat
from breezy.errors import IncompatibleFormat
from breezy.revision import NULL_REVISION
from breezy.tests import TestCaseWithTransport
class ForeignBranchFactory:
    """Factory of branches for ForeignBranchTests."""

    def make_empty_branch(self, transport):
        """Create an empty branch with no commits in it."""
        raise NotImplementedError(self.make_empty_branch)

    def make_branch(self, transport):
        """Create *some* branch, may be empty or not."""
        return self.make_empty_branch(transport)