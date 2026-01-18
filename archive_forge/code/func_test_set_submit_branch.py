from breezy import branch, errors, tests
from breezy.tests import per_branch
def test_set_submit_branch(self):
    b = self.make_branch('.')
    b.set_submit_branch('foo')
    b = branch.Branch.open('.')
    self.assertEqual('foo', b.get_submit_branch())