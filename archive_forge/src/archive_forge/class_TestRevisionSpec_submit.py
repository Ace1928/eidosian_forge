import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class TestRevisionSpec_submit(TestRevisionSpec):

    def test_submit_branch(self):
        self.assertRaises(errors.NoSubmitBranch, self.get_in_history, 'submit:')
        self.tree.branch.set_parent('../tree2')
        self.assertInHistoryIs(None, b'alt_r2', 'submit:')
        self.tree.branch.set_parent('bogus')
        self.assertRaises(errors.NotBranchError, self.get_in_history, 'submit:')
        self.tree.branch.set_submit_branch('tree2')
        self.assertInHistoryIs(None, b'alt_r2', 'submit:')

    def test_as_revision_id(self):
        self.tree.branch.set_submit_branch('tree2')
        self.assertAsRevisionId(b'alt_r2', 'branch:tree2')