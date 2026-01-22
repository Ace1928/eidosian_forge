import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class TestRevisionSpec_mainline(TestRevisionSpec):

    def test_as_revision_id(self):
        self.assertAsRevisionId(b'r1', 'mainline:1')
        self.assertAsRevisionId(b'r2', 'mainline:1.1.1')
        self.assertAsRevisionId(b'r2', 'mainline:revid:alt_r2')
        spec = RevisionSpec.from_string('mainline:revid:alt_r22')
        e = self.assertRaises(InvalidRevisionSpec, spec.as_revision_id, self.tree.branch)
        self.assertContainsRe(str(e), "Requested revision: 'mainline:revid:alt_r22' does not exist in branch: ")

    def test_in_history(self):
        self.assertInHistoryIs(2, b'r2', 'mainline:revid:alt_r2')