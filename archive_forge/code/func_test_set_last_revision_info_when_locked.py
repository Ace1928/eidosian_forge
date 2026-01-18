from breezy.tests.per_branch import TestCaseWithBranch
def test_set_last_revision_info_when_locked(self):
    """Calling set_last_revision_info should reset the cache."""
    branch, revmap, calls = self.get_instrumented_branch()
    with branch.lock_write():
        self.assertEqual({revmap['1']: (1,), revmap['2']: (2,), revmap['3']: (3,), revmap['1.1.1']: (1, 1, 1)}, branch.get_revision_id_to_revno_map())
        branch.set_last_revision_info(2, revmap['2'])
        self.assertEqual({revmap['1']: (1,), revmap['2']: (2,)}, branch.get_revision_id_to_revno_map())
        self.assertEqual({revmap['1']: (1,), revmap['2']: (2,)}, branch.get_revision_id_to_revno_map())
        self.assertEqual(['_gen_revno_map'] * 2, calls)