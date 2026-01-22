import unittest
import gyp.MSVSSettings as MSVSSettings
from io import StringIO
def _ExpectedWarnings(self, expected):
    """Compares recorded lines to expected warnings."""
    self.stderr.seek(0)
    actual = self.stderr.read().split('\n')
    actual = [line for line in actual if line]
    self.assertEqual(sorted(expected), sorted(actual))