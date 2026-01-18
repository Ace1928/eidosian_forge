import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_yesterday(self):
    self.assertInHistoryIs(1, self.revid1, 'date:yesterday')