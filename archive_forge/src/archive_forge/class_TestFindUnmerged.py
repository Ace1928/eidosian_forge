from .. import missing, tests
from ..missing import iter_log_revisions
from . import TestCaseWithTransport
class TestFindUnmerged(tests.TestCaseWithTransport):

    def assertUnmerged(self, local, remote, local_branch, remote_branch, restrict='all', include_merged=False, backward=False, local_revid_range=None, remote_revid_range=None):
        """Check the output of find_unmerged_mainline_revisions"""
        local_extra, remote_extra = missing.find_unmerged(local_branch, remote_branch, restrict, include_merged=include_merged, backward=backward, local_revid_range=local_revid_range, remote_revid_range=remote_revid_range)
        self.assertEqual(local, local_extra)
        self.assertEqual(remote, remote_extra)

    def test_same_branch(self):
        tree = self.make_branch_and_tree('tree')
        rev1 = tree.commit('one')
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertUnmerged([], [], tree.branch, tree.branch)
        self.assertUnmerged([], [], tree.branch, tree.branch, local_revid_range=(rev1, rev1))

    def test_one_ahead(self):
        tree = self.make_branch_and_tree('tree')
        tree.commit('one')
        tree2 = tree.controldir.sprout('tree2').open_workingtree()
        rev2 = tree2.commit('two')
        self.assertUnmerged([], [('2', rev2, 0)], tree.branch, tree2.branch)
        self.assertUnmerged([('2', rev2, 0)], [], tree2.branch, tree.branch)

    def test_restrict(self):
        tree = self.make_branch_and_tree('tree')
        tree.commit('one')
        tree2 = tree.controldir.sprout('tree2').open_workingtree()
        rev2 = tree2.commit('two')
        self.assertUnmerged([], [('2', rev2, 0)], tree.branch, tree2.branch)
        self.assertUnmerged([], None, tree.branch, tree2.branch, 'local')
        self.assertUnmerged(None, [('2', rev2, 0)], tree.branch, tree2.branch, 'remote')

    def test_merged(self):
        tree = self.make_branch_and_tree('tree')
        rev1 = tree.commit('one')
        tree2 = tree.controldir.sprout('tree2').open_workingtree()
        tree2.commit('two')
        tree2.commit('three')
        tree.merge_from_branch(tree2.branch)
        rev4 = tree.commit('four')
        self.assertUnmerged([('2', rev4, 0)], [], tree.branch, tree2.branch)
        self.assertUnmerged([('2', rev4, 0)], [], tree.branch, tree2.branch, local_revid_range=(rev4, rev4))
        self.assertUnmerged([], [], tree.branch, tree2.branch, local_revid_range=(rev1, rev1))

    def test_include_merged(self):
        tree = self.make_branch_and_tree('tree')
        tree.commit('one', rev_id=b'rev1')
        tree2 = tree.controldir.sprout('tree2').open_workingtree()
        tree2.commit('two', rev_id=b'rev2')
        rev3 = tree2.commit('three', rev_id=b'rev3')
        tree3 = tree2.controldir.sprout('tree3').open_workingtree()
        rev4 = tree3.commit('four', rev_id=b'rev4')
        rev5 = tree3.commit('five', rev_id=b'rev5')
        tree2.merge_from_branch(tree3.branch)
        rev6 = tree2.commit('six', rev_id=b'rev6')
        self.assertUnmerged([], [('2', b'rev2', 0), ('3', b'rev3', 0), ('4', b'rev6', 0), ('3.1.1', b'rev4', 1), ('3.1.2', b'rev5', 1)], tree.branch, tree2.branch, include_merged=True)
        self.assertUnmerged([], [('4', b'rev6', 0), ('3.1.2', b'rev5', 1), ('3.1.1', b'rev4', 1), ('3', b'rev3', 0), ('2', b'rev2', 0)], tree.branch, tree2.branch, include_merged=True, backward=True)
        self.assertUnmerged([], [('4', b'rev6', 0)], tree.branch, tree2.branch, include_merged=True, remote_revid_range=(rev6, rev6))
        self.assertUnmerged([], [('3', b'rev3', 0), ('3.1.1', b'rev4', 1)], tree.branch, tree2.branch, include_merged=True, remote_revid_range=(rev3, rev4))
        self.assertUnmerged([], [('4', b'rev6', 0), ('3.1.2', b'rev5', 1)], tree.branch, tree2.branch, include_merged=True, remote_revid_range=(rev5, rev6))

    def test_revision_range(self):
        local = self.make_branch_and_tree('local')
        lrevid1 = local.commit('one')
        remote = local.controldir.sprout('remote').open_workingtree()
        rrevid2 = remote.commit('two')
        rrevid3 = remote.commit('three')
        rrevid4 = remote.commit('four')
        lrevid2 = local.commit('two')
        lrevid3 = local.commit('three')
        lrevid4 = local.commit('four')
        local_extra = [('2', lrevid2, 0), ('3', lrevid3, 0), ('4', lrevid4, 0)]
        remote_extra = [('2', rrevid2, 0), ('3', rrevid3, 0), ('4', rrevid4, 0)]
        self.assertUnmerged(local_extra, remote_extra, local.branch, remote.branch)
        self.assertUnmerged(local_extra, remote_extra, local.branch, remote.branch, local_revid_range=(None, None), remote_revid_range=(None, None))
        self.assertUnmerged([('2', lrevid2, 0)], remote_extra, local.branch, remote.branch, local_revid_range=(lrevid2, lrevid2))
        self.assertUnmerged([('2', lrevid2, 0), ('3', lrevid3, 0)], remote_extra, local.branch, remote.branch, local_revid_range=(lrevid2, lrevid3))
        self.assertUnmerged([('2', lrevid2, 0), ('3', lrevid3, 0)], None, local.branch, remote.branch, 'local', local_revid_range=(lrevid2, lrevid3))
        self.assertUnmerged(local_extra, [('2', rrevid2, 0)], local.branch, remote.branch, remote_revid_range=(None, rrevid2))
        self.assertUnmerged(local_extra, [('2', rrevid2, 0)], local.branch, remote.branch, remote_revid_range=(lrevid1, rrevid2))
        self.assertUnmerged(local_extra, [('2', rrevid2, 0)], local.branch, remote.branch, remote_revid_range=(rrevid2, rrevid2))
        self.assertUnmerged(local_extra, [('2', rrevid2, 0), ('3', rrevid3, 0)], local.branch, remote.branch, remote_revid_range=(None, rrevid3))
        self.assertUnmerged(local_extra, [('2', rrevid2, 0), ('3', rrevid3, 0)], local.branch, remote.branch, remote_revid_range=(rrevid2, rrevid3))
        self.assertUnmerged(local_extra, [('3', rrevid3, 0)], local.branch, remote.branch, remote_revid_range=(rrevid3, rrevid3))
        self.assertUnmerged(None, [('2', rrevid2, 0), ('3', rrevid3, 0)], local.branch, remote.branch, 'remote', remote_revid_range=(rrevid2, rrevid3))
        self.assertUnmerged([('3', lrevid3, 0)], [('3', rrevid3, 0)], local.branch, remote.branch, local_revid_range=(lrevid3, lrevid3), remote_revid_range=(rrevid3, rrevid3))