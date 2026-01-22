import warnings
from breezy import bugtracker, revision
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tests.matchers import MatchesAncestry
class TestIsAncestor(TestCaseWithTransport):

    def test_recorded_ancestry(self):
        """Test that commit records all ancestors"""
        br1, br2 = make_branches(self)
        d = [(b'a@u-0-0', [b'a@u-0-0']), (b'a@u-0-1', [b'a@u-0-0', b'a@u-0-1']), (b'a@u-0-2', [b'a@u-0-0', b'a@u-0-1', b'a@u-0-2']), (b'b@u-0-3', [b'a@u-0-0', b'a@u-0-1', b'a@u-0-2', b'b@u-0-3']), (b'b@u-0-4', [b'a@u-0-0', b'a@u-0-1', b'a@u-0-2', b'b@u-0-3', b'b@u-0-4']), (b'a@u-0-3', [b'a@u-0-0', b'a@u-0-1', b'a@u-0-2', b'b@u-0-3', b'b@u-0-4', b'a@u-0-3']), (b'a@u-0-4', [b'a@u-0-0', b'a@u-0-1', b'a@u-0-2', b'b@u-0-3', b'b@u-0-4', b'a@u-0-3', b'a@u-0-4']), (b'b@u-0-5', [b'a@u-0-0', b'a@u-0-1', b'a@u-0-2', b'b@u-0-3', b'b@u-0-4', b'b@u-0-5']), (b'a@u-0-5', [b'a@u-0-0', b'a@u-0-1', b'a@u-0-2', b'a@u-0-3', b'a@u-0-4', b'b@u-0-3', b'b@u-0-4', b'b@u-0-5', b'a@u-0-5']), (b'b@u-0-6', [b'a@u-0-0', b'a@u-0-1', b'a@u-0-2', b'a@u-0-4', b'b@u-0-3', b'b@u-0-4', b'b@u-0-5', b'b@u-0-6'])]
        br1_only = (b'a@u-0-3', b'a@u-0-4', b'a@u-0-5')
        br2_only = (b'b@u-0-6',)
        for branch in (br1, br2):
            for rev_id, anc in d:
                if rev_id in br1_only and branch is not br1:
                    continue
                if rev_id in br2_only and branch is not br2:
                    continue
                self.assertThat(anc, MatchesAncestry(branch.repository, rev_id))