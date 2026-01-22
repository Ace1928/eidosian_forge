import sys
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy import tests, transform
from breezy.bzr import inventory, remote
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestFileIdInvolvedSuperset(FileIdInvolvedBase):
    scenarios = all_repository_vf_format_scenarios()

    def setUp(self):
        super().setUp()
        self.branch = None
        main_wt = self.make_branch_and_tree('main')
        main_branch = main_wt.branch
        self.build_tree(['main/a', 'main/b', 'main/c'])
        main_wt.add(['a', 'b', 'c'], ids=[b'a-file-id-2006-01-01-abcd', b'b-file-id-2006-01-01-defg', b'c-funky<file-id>quiji\'"%bo'])
        try:
            main_wt.commit('Commit one', rev_id=b'rev-A')
        except errors.IllegalPath:
            if sys.platform == 'win32':
                raise tests.TestSkipped('Old repository formats do not support file ids with <> on win32')
            raise
        branch2_wt = self.make_branch_and_tree('branch2')
        branch2_wt.pull(main_branch)
        branch2_bzrdir = branch2_wt.controldir
        branch2_branch = branch2_bzrdir.open_branch()
        set_executability(branch2_wt, 'b', True)
        branch2_wt.commit('branch2, Commit one', rev_id=b'rev-J')
        main_wt.merge_from_branch(branch2_branch)
        set_executability(main_wt, 'b', False)
        main_wt.commit('merge branch1, rev-22', rev_id=b'rev-G')
        self.branch = main_branch

    def test_fileid_involved_full_compare2(self):
        self.branch.lock_read()
        self.addCleanup(self.branch.unlock)
        graph = self.branch.repository.get_graph()
        history = list(graph.iter_lefthand_ancestry(self.branch.last_revision(), [_mod_revision.NULL_REVISION]))
        history.reverse()
        old_rev = history[0]
        new_rev = history[1]
        unique_revs = graph.find_unique_ancestors(new_rev, [old_rev])
        l1 = self.branch.repository.fileids_altered_by_revision_ids(unique_revs)
        l1 = set(l1.keys())
        l2 = self.compare_tree_fileids(self.branch, old_rev, new_rev)
        self.assertNotEqual(l2, l1)
        self.assertSubset(l2, l1)