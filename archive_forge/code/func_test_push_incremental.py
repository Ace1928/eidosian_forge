from breezy.tests.per_controldir import TestCaseWithControlDir
from ...controldir import NoColocatedBranchSupport
from ...errors import LossyPushToSameVCS, NoSuchRevision, TagsNotSupported
from ...revision import NULL_REVISION
from .. import TestNotApplicable
def test_push_incremental(self):
    tree, rev1 = self.create_simple_tree()
    dir = self.make_repository('dir').controldir
    dir.push_branch(tree.branch)
    self.build_tree(['tree/b'])
    tree.add(['b'])
    rev_2 = tree.commit('two')
    result = dir.push_branch(tree.branch)
    self.assertEqual(tree.last_revision(), result.branch_push_result.new_revid)
    self.assertEqual(2, result.branch_push_result.new_revno)
    self.assertEqual(tree.branch.base, result.source_branch.base)
    self.assertEqual(dir.open_branch().base, result.target_branch.base)