import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
def test_unhashable(self):
    """We should get a proper exception here."""
    self.assertRaises(TypeError, self._PatienceSequenceMatcher, None, [[]], [])
    self.assertRaises(TypeError, self._PatienceSequenceMatcher, None, ['valid', []], [])
    self.assertRaises(TypeError, self._PatienceSequenceMatcher, None, ['valid'], [[]])
    self.assertRaises(TypeError, self._PatienceSequenceMatcher, None, ['valid'], ['valid', []])