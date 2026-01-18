from ..branchbuilder import BranchBuilder
from . import TestCaseWithMemoryTransport
from .matchers import MatchesAncestry
def test_straightline_ancestry(self):
    """Test ancestry file when just committing."""
    builder = BranchBuilder(self.get_transport())
    rev_id_one = builder.build_commit()
    rev_id_two = builder.build_commit()
    branch = builder.get_branch()
    self.assertThat([rev_id_one, rev_id_two], MatchesAncestry(branch.repository, rev_id_two))
    self.assertThat([rev_id_one], MatchesAncestry(branch.repository, rev_id_one))