import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class RevisionSpecMatchOnTrap(RevisionSpec):

    def _match_on(self, branch, revs):
        self.last_call = (branch, revs)
        return super()._match_on(branch, revs)