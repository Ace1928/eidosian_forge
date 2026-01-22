from breezy.branch import UnstackableBranchFormat
from breezy.errors import IncompatibleFormat
from breezy.revision import NULL_REVISION
from breezy.tests import TestCaseWithTransport
class ForeignBranchFormatTests(TestCaseWithTransport):
    """Basic tests for foreign branch format objects."""
    branch_format = None

    def test_initialize(self):
        """Test this format is not initializable.

        Remote branches may be initializable on their own, but none currently
        support living in .bzr/branch.
        """
        bzrdir = self.make_controldir('dir')
        self.assertRaises(IncompatibleFormat, self.branch_format.initialize, bzrdir)

    def test_get_format_description_type(self):
        self.assertIsInstance(self.branch_format.get_format_description(), str)

    def test_network_name(self):
        self.assertIsInstance(self.branch_format.network_name(), bytes)