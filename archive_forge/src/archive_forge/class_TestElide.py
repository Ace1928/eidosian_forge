import sys
import unittest
import os
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from IPython.testing import tools as tt
from IPython.terminal.ptutils import _elide, _adjust_completion_text_based_on_context
from IPython.terminal.shortcuts.auto_suggest import NavigableAutoSuggestFromHistory
class TestElide(unittest.TestCase):

    def test_elide(self):
        _elide('concatenate((a1, a2, ...), axis', '')
        _elide('concatenate((a1, a2, ..), . axis', '')
        self.assertEqual(_elide('aaaa.bbbb.ccccc.dddddd.eeeee.fffff.gggggg.hhhhhh', ''), 'aaaa.b…g.hhhhhh')
        test_string = os.sep.join(['', 10 * 'a', 10 * 'b', 10 * 'c', ''])
        expect_string = os.sep + 'a' + '…' + 'b' + os.sep + 10 * 'c'
        self.assertEqual(_elide(test_string, ''), expect_string)

    def test_elide_typed_normal(self):
        self.assertEqual(_elide('the quick brown fox jumped over the lazy dog', 'the quick brown fox', min_elide=10), 'the…fox jumped over the lazy dog')

    def test_elide_typed_short_match(self):
        """
        if the match is too short we don't elide.
        avoid the "the...the"
        """
        self.assertEqual(_elide('the quick brown fox jumped over the lazy dog', 'the', min_elide=10), 'the quick brown fox jumped over the lazy dog')

    def test_elide_typed_no_match(self):
        """
        if the match is too short we don't elide.
        avoid the "the...the"
        """
        self.assertEqual(_elide('the quick brown fox jumped over the lazy dog', 'the quick red fox', min_elide=10), 'the quick brown fox jumped over the lazy dog')