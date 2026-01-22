import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def assertInHistoryIs(self, exp_revno, exp_revision_id, revision_spec):
    rev_info = self.get_in_history(revision_spec)
    self.assertEqual(exp_revno, rev_info.revno, 'Revision spec: %r returned wrong revno: %r != %r' % (revision_spec, exp_revno, rev_info.revno))
    self.assertEqual(exp_revision_id, rev_info.rev_id, 'Revision spec: %r returned wrong revision id: %r != %r' % (revision_spec, exp_revision_id, rev_info.rev_id))