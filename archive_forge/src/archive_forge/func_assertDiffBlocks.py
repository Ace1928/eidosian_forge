import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
def assertDiffBlocks(self, a, b, expected_blocks):
    """Check that the sequence matcher returns the correct blocks.

        :param a: A sequence to match
        :param b: Another sequence to match
        :param expected_blocks: The expected output, not including the final
            matching block (len(a), len(b), 0)
        """
    matcher = self._PatienceSequenceMatcher(None, a, b)
    blocks = matcher.get_matching_blocks()
    last = blocks.pop()
    self.assertEqual((len(a), len(b), 0), last)
    self.assertEqual(expected_blocks, blocks)