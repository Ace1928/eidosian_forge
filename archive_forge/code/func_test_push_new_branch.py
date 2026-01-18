from breezy.tests.per_controldir import TestCaseWithControlDir
from ...controldir import NoColocatedBranchSupport
from ...errors import LossyPushToSameVCS, NoSuchRevision, TagsNotSupported
from ...revision import NULL_REVISION
from .. import TestNotApplicable
def test_push_new_branch(self):
    tree, rev_1 = self.create_simple_tree()
    dir = self.make_repository('dir').controldir
    result = dir.push_branch(tree.branch)
    self.assertEqual(tree.branch, result.source_branch)
    self.assertEqual(dir.open_branch().base, result.target_branch.base)
    self.assertEqual(dir.open_branch().base, tree.branch.get_push_location())