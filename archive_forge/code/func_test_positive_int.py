import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_positive_int(self):
    self.assertInHistoryIs(0, b'null:', '0')
    self.assertInHistoryIs(1, b'r1', '1')
    self.assertInHistoryIs(2, b'r2', '2')
    self.assertInvalid('3')