from .. import missing, tests
from ..missing import iter_log_revisions
from . import TestCaseWithTransport
def test_one_ahead(self):
    tree = self.make_branch_and_tree('tree')
    tree.commit('one')
    tree2 = tree.controldir.sprout('tree2').open_workingtree()
    rev2 = tree2.commit('two')
    self.assertUnmerged([], [('2', rev2, 0)], tree.branch, tree2.branch)
    self.assertUnmerged([('2', rev2, 0)], [], tree2.branch, tree.branch)