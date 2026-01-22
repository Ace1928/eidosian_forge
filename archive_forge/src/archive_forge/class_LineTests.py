from io import StringIO
from twisted.python import text
from twisted.trial import unittest
class LineTests(unittest.TestCase):
    """
    Tests for L{isMultiline} and L{endsInNewline}.
    """

    def test_isMultiline(self) -> None:
        """
        L{text.isMultiline} returns C{True} if the string has a newline in it.
        """
        s = 'This code\n "breaks."'
        m = text.isMultiline(s)
        self.assertTrue(m)
        s = 'This code does not "break."'
        m = text.isMultiline(s)
        self.assertFalse(m)

    def test_endsInNewline(self) -> None:
        """
        L{text.endsInNewline} returns C{True} if the string ends in a newline.
        """
        s = 'newline\n'
        m = text.endsInNewline(s)
        self.assertTrue(m)
        s = 'oldline'
        m = text.endsInNewline(s)
        self.assertFalse(m)