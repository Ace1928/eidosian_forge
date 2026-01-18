from ..memorybranch import MemoryBranch
from . import TestCaseWithTransport
def test_revision_id_to_revno(self):
    self.assertEqual(2, self.branch.revision_id_to_revno(self.revid2))
    self.assertEqual(1, self.branch.revision_id_to_revno(self.revid1))