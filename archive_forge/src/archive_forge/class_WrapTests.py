from io import StringIO
from twisted.python import text
from twisted.trial import unittest
class WrapTests(unittest.TestCase):
    """
    Tests for L{text.greedyWrap}.
    """

    def setUp(self) -> None:
        self.lineWidth = 72
        self.sampleSplitText = sampleText.split()
        self.output = text.wordWrap(sampleText, self.lineWidth)

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

    def test_wordMatch(self) -> None:
        """
        Compare the lists of words.
        """
        words = []
        for line in self.output:
            words.extend(line.split())
        self.assertTrue(self.sampleSplitText == words)

    def test_lineLength(self) -> None:
        """
        Check the length of the lines.
        """
        failures = []
        for line in self.output:
            if not len(line) <= self.lineWidth:
                failures.append(len(line))
        if failures:
            self.fail('%d of %d lines were too long.\n%d < %s' % (len(failures), len(self.output), self.lineWidth, failures))

    def test_doubleNewline(self) -> None:
        """
        Allow paragraphs delimited by two 
s.
        """
        sampleText = 'et\n\nphone\nhome.'
        result = text.wordWrap(sampleText, self.lineWidth)
        self.assertEqual(result, ['et', '', 'phone home.', ''])