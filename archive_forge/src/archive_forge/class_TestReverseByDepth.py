import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestReverseByDepth(tests.TestCase):
    """Test reverse_by_depth behavior.

    This is used to present revisions in forward (oldest first) order in a nice
    layout.

    The tests use lighter revision description to ease reading.
    """

    def assertReversed(self, forward, backward):

        def complete_revisions(l):
            """Transform the description to suit the API.

            Tests use (revno, depth) whil the API expects (revid, revno, depth).
            Since the revid is arbitrary, we just duplicate revno
            """
            return [(r, r, d) for r, d in l]
        forward = complete_revisions(forward)
        backward = complete_revisions(backward)
        self.assertEqual(forward, log.reverse_by_depth(backward))

    def test_mainline_revisions(self):
        self.assertReversed([('1', 0), ('2', 0)], [('2', 0), ('1', 0)])

    def test_merged_revisions(self):
        self.assertReversed([('1', 0), ('2', 0), ('2.2', 1), ('2.1', 1)], [('2', 0), ('2.1', 1), ('2.2', 1), ('1', 0)])

    def test_shifted_merged_revisions(self):
        """Test irregular layout.

        Requesting revisions touching a file can produce "holes" in the depths.
        """
        self.assertReversed([('1', 0), ('2', 0), ('1.1', 2), ('1.2', 2)], [('2', 0), ('1.2', 2), ('1.1', 2), ('1', 0)])

    def test_merged_without_child_revisions(self):
        """Test irregular layout.

        Revision ranges can produce "holes" in the depths.
        """
        self.assertReversed([('1', 2), ('2', 2), ('3', 3), ('4', 4)], [('4', 4), ('3', 3), ('2', 2), ('1', 2)])
        self.assertReversed([('1', 2), ('2', 2), ('3', 3), ('4', 4)], [('3', 3), ('4', 4), ('2', 2), ('1', 2)])