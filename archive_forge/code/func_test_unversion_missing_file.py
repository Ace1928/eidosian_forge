from breezy import errors, transport
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_unversion_missing_file(self):
    """WT.unversion(['missing']) raises NoSuchId."""
    tree = self.make_branch_and_tree('.')
    self.assertRaises(transport.NoSuchFile, tree.unversion, ['missing'])