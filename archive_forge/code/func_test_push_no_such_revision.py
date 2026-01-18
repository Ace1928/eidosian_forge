from breezy.tests.per_controldir import TestCaseWithControlDir
from ...controldir import NoColocatedBranchSupport
from ...errors import LossyPushToSameVCS, NoSuchRevision, TagsNotSupported
from ...revision import NULL_REVISION
from .. import TestNotApplicable
def test_push_no_such_revision(self):
    tree, rev_1 = self.create_simple_tree()
    dir = self.make_repository('dir').controldir
    self.assertRaises(NoSuchRevision, dir.push_branch, tree.branch, revision_id=b'idonotexist')