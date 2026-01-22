from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
class TestMergeFetch(TestCaseWithTransport):

    def test_merge_fetches_unrelated(self):
        """Merge brings across history from unrelated source"""
        wt1 = self.make_branch_and_tree('br1')
        br1 = wt1.branch
        wt1.commit(message='rev 1-1', rev_id=b'1-1')
        wt1.commit(message='rev 1-2', rev_id=b'1-2')
        wt2 = self.make_branch_and_tree('br2')
        br2 = wt2.branch
        wt2.commit(message='rev 2-1', rev_id=b'2-1')
        wt2.merge_from_branch(br1, from_revision=b'null:')
        self._check_revs_present(br2)

    def test_merge_fetches(self):
        """Merge brings across history from source"""
        wt1 = self.make_branch_and_tree('br1')
        br1 = wt1.branch
        wt1.commit(message='rev 1-1', rev_id=b'1-1')
        dir_2 = br1.controldir.sprout('br2')
        br2 = dir_2.open_branch()
        wt1.commit(message='rev 1-2', rev_id=b'1-2')
        wt2 = dir_2.open_workingtree()
        wt2.commit(message='rev 2-1', rev_id=b'2-1')
        wt2.merge_from_branch(br1)
        self._check_revs_present(br2)

    def _check_revs_present(self, br2):
        for rev_id in [b'1-1', b'1-2', b'2-1']:
            self.assertTrue(br2.repository.has_revision(rev_id))
            rev = br2.repository.get_revision(rev_id)
            self.assertEqual(rev.revision_id, rev_id)
            self.assertTrue(br2.repository.get_inventory(rev_id))