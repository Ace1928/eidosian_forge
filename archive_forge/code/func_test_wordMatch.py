from io import StringIO
from twisted.python import text
from twisted.trial import unittest
def test_wordMatch(self) -> None:
    """
        Compare the lists of words.
        """
    words = []
    for line in self.output:
        words.extend(line.split())
    self.assertTrue(self.sampleSplitText == words)