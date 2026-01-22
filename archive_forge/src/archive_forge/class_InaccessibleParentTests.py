from breezy import branch, errors
from breezy.tests import per_branch, test_server
class InaccessibleParentTests(per_branch.TestCaseWithBranch):
    """Tests with branches with "inaccessible" parents.

    An "inaccessible" parent location is one that cannot be represented, e.g. if
    a child branch says its parent is at "../../my-parent", but that child is at
    "http://host/one" then that parent location is inaccessible.  These
    branches' get_parent method will raise InaccessibleParent.
    """

    def setUp(self):
        super().setUp()
        if self.transport_server in (test_server.LocalURLServer, None):
            self.transport_readonly_server = test_server.TestingChrootServer

    def get_branch_with_invalid_parent(self):
        """Get a branch whose get_parent will raise InaccessibleParent."""
        self.build_tree(['parent/', 'parent/path/', 'parent/path/to/', 'child/', 'child/path/', 'child/path/to/'], transport=self.get_transport())
        self.make_branch('parent/path/to/a').controldir.sprout(self.get_url('child/path/to/b'))
        self.get_transport().rename('child/path/to/b', 'b')
        branch_b = branch.Branch.open(self.get_readonly_url('b'))
        return branch_b

    def test_get_parent_invalid(self):
        branch_b = self.get_branch_with_invalid_parent()
        self.assertRaises(errors.InaccessibleParent, branch_b.get_parent)

    def test_clone_invalid_parent(self):
        branch_b = self.get_branch_with_invalid_parent()
        branch_c = branch_b.controldir.clone('c').open_branch()
        self.assertEqual(None, branch_c.get_parent())

    def test_sprout_invalid_parent(self):
        branch_b = self.get_branch_with_invalid_parent()
        branch_c = branch_b.controldir.sprout('c').open_branch()
        self.assertEqual(branch_b.base, branch_c.get_parent())