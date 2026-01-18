import os
from breezy import branch, conflicts, controldir, errors, mutabletree, osutils
from breezy import revision as _mod_revision
from breezy import tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.commit import CannotCommitSelectedFileMerge, PointlessCommit
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.tests.testui import ProgressRecordingUIFactory
def test_record_initial_ghost(self):
    """The working tree needs to record ghosts during commit."""
    wt = self.make_branch_and_tree('.')
    if not wt.branch.repository._format.supports_ghosts:
        raise tests.TestNotApplicable('format does not support ghosts')
    wt.set_parent_ids([b'non:existent@rev--ision--0--2'], allow_leftmost_as_ghost=True)
    rev_id = wt.commit('commit against a ghost first parent.')
    rev = wt.branch.repository.get_revision(rev_id)
    self.assertEqual(rev.parent_ids, [b'non:existent@rev--ision--0--2'])
    self.assertEqual(len(rev.parent_sha1s), 0)