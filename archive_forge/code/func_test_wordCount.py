from io import StringIO
from twisted.python import text
from twisted.trial import unittest
def test_wordCount(self) -> None:
    """
        Compare the number of words.
        """
    words = []
    for line in self.output:
        words.extend(line.split())
    wordCount = len(words)
    sampleTextWordCount = len(self.sampleSplitText)
    self.assertEqual(wordCount, sampleTextWordCount)