from twisted.python import usage
from twisted.trial import unittest
def test_whitespaceStripFlagsAndParameters(self):
    """
        Extra whitespace in flag and parameters docs is stripped.
        """
    lines = [s for s in str(self.nice).splitlines() if s.find('aflag') >= 0]
    self.assertTrue(len(lines) > 0)
    self.assertTrue(lines[0].find('flagallicious') >= 0)