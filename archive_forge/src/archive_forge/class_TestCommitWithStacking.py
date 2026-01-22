from breezy import errors, tests, urlutils
from breezy.bzr import remote
from breezy.tests.per_repository import TestCaseWithRepository
class TestCommitWithStacking(TestCaseWithStackedTarget):

    def setUp(self):
        super().setUp()
        format = self.repository_format
        if not (isinstance(format, remote.RemoteRepositoryFormat) or format.supports_chks):
            raise tests.TestNotApplicable('stacked commit only supported for chk repositories')

    def get_only_repo(self, tree):
        """Open just the repository used by this tree.

        This returns a read locked Repository object without any stacking
        fallbacks.
        """
        repo = tree.branch.repository.controldir.open_repository()
        repo.lock_read()
        self.addCleanup(repo.unlock)
        return repo

    def assertPresent(self, expected, vf, keys):
        """Check which of the supplied keys are present."""
        parent_map = vf.get_parent_map(keys)
        self.assertEqual(sorted(expected), sorted(parent_map))

    def test_simple_commit(self):
        base_tree, stacked_tree = self.make_stacked_target()
        self.assertEqual(1, len(stacked_tree.branch.repository._fallback_repositories))
        self.build_tree_contents([('stacked/f1.txt', b'new content\n')])
        stacked_tree.commit('new content', rev_id=b'new-rev-id')
        stacked_only_repo = self.get_only_repo(stacked_tree)
        self.assertPresent([self.r2_key], stacked_only_repo.inventories, [self.r1_key, self.r2_key])
        stacked2_branch = base_tree.controldir.sprout('stacked2', stacked=True).open_branch()
        stacked2_branch.repository.fetch(stacked_only_repo, revision_id=b'new-rev-id')

    def test_merge_commit(self):
        base_tree, stacked_tree = self.make_stacked_target()
        self.build_tree_contents([('base/f1.txt', b'new content\n')])
        r3_key = (b'rev3-id',)
        base_tree.commit('second base', rev_id=r3_key[0])
        to_be_merged_tree = base_tree.controldir.sprout('merged').open_workingtree()
        self.build_tree(['merged/f2.txt'])
        to_be_merged_tree.add(['f2.txt'], ids=[b'f2.txt-id'])
        to_merge_key = (b'to-merge-rev-id',)
        to_be_merged_tree.commit('new-to-be-merged', rev_id=to_merge_key[0])
        stacked_tree.merge_from_branch(to_be_merged_tree.branch)
        merged_key = (b'merged-rev-id',)
        stacked_tree.commit('merge', rev_id=merged_key[0])
        stacked_only_repo = self.get_only_repo(stacked_tree)
        all_keys = [self.r1_key, self.r2_key, r3_key, to_merge_key, merged_key]
        self.assertPresent([to_merge_key, merged_key], stacked_only_repo.revisions, all_keys)
        self.assertPresent([self.r2_key, r3_key, to_merge_key, merged_key], stacked_only_repo.inventories, all_keys)

    def test_merge_from_master(self):
        base_tree, stacked_tree = self.make_stacked_target()
        self.build_tree_contents([('base/f1.txt', b'new content\n')])
        r3_key = (b'rev3-id',)
        base_tree.commit('second base', rev_id=r3_key[0])
        stacked_tree.merge_from_branch(base_tree.branch)
        merged_key = (b'merged-rev-id',)
        stacked_tree.commit('merge', rev_id=merged_key[0])
        all_keys = [self.r1_key, self.r2_key, r3_key, merged_key]
        stacked_only_repo = self.get_only_repo(stacked_tree)
        self.assertPresent([merged_key], stacked_only_repo.revisions, all_keys)
        self.assertPresent([self.r2_key, r3_key, merged_key], stacked_only_repo.inventories, all_keys)

    def test_multi_stack(self):
        """base + stacked + stacked-on-stacked"""
        base_tree, stacked_tree = self.make_stacked_target()
        self.build_tree(['stacked/f3.txt'])
        stacked_tree.add(['f3.txt'], ids=[b'f3.txt-id'])
        stacked_key = (b'stacked-rev-id',)
        stacked_tree.commit('add f3', rev_id=stacked_key[0])
        stacked_only_repo = self.get_only_repo(stacked_tree)
        self.assertPresent([self.r2_key], stacked_only_repo.inventories, [self.r1_key, self.r2_key])
        stacked2_url = urlutils.join(base_tree.branch.base, '../stacked2')
        stacked2_bzrdir = stacked_tree.controldir.sprout(stacked2_url, revision_id=self.r1_key[0], stacked=True)
        if isinstance(stacked2_bzrdir, remote.RemoteBzrDir):
            stacked2_branch = stacked2_bzrdir.open_branch()
            stacked2_tree = stacked2_branch.create_checkout('stacked2', lightweight=True)
        else:
            stacked2_tree = stacked2_bzrdir.open_workingtree()
        self.build_tree(['stacked2/f3.txt'])
        stacked2_only_repo = self.get_only_repo(stacked2_tree)
        self.assertPresent([], stacked2_only_repo.inventories, [self.r1_key, self.r2_key])
        stacked2_tree.add(['f3.txt'], ids=[b'f3.txt-id'])
        stacked2_tree.commit('add f3', rev_id=b'stacked2-rev-id')
        stacked2_only_repo.refresh_data()
        self.assertPresent([self.r1_key], stacked2_only_repo.inventories, [self.r1_key, self.r2_key])

    def test_commit_with_ghosts_fails(self):
        base_tree, stacked_tree = self.make_stacked_target()
        stacked_tree.set_parent_ids([stacked_tree.last_revision(), b'ghost-rev-id'])
        self.assertRaises(errors.BzrError, stacked_tree.commit, 'failed_commit')

    def test_commit_with_ghost_in_ancestry(self):
        base_tree, stacked_tree = self.make_stacked_target()
        self.build_tree_contents([('base/f1.txt', b'new content\n')])
        r3_key = (b'rev3-id',)
        base_tree.commit('second base', rev_id=r3_key[0])
        to_be_merged_tree = base_tree.controldir.sprout('merged').open_workingtree()
        self.build_tree(['merged/f2.txt'])
        to_be_merged_tree.add(['f2.txt'], ids=[b'f2.txt-id'])
        ghost_key = (b'ghost-rev-id',)
        to_be_merged_tree.set_parent_ids([r3_key[0], ghost_key[0]])
        to_merge_key = (b'to-merge-rev-id',)
        to_be_merged_tree.commit('new-to-be-merged', rev_id=to_merge_key[0])
        stacked_tree.merge_from_branch(to_be_merged_tree.branch)
        merged_key = (b'merged-rev-id',)
        stacked_tree.commit('merge', rev_id=merged_key[0])
        stacked_only_repo = self.get_only_repo(stacked_tree)
        all_keys = [self.r1_key, self.r2_key, r3_key, to_merge_key, merged_key, ghost_key]
        self.assertPresent([to_merge_key, merged_key], stacked_only_repo.revisions, all_keys)
        self.assertPresent([self.r2_key, r3_key, to_merge_key, merged_key], stacked_only_repo.inventories, all_keys)