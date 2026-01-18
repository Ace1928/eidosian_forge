from twisted.conch.insults import text
from twisted.conch.insults.text import attributes as A
from twisted.trial import unittest
def test_underline(self) -> None:
    """
        The underline formatting attribute, L{A.underline}, emits the VT102
        control sequence to enable underlining when flattened.
        """
    self.assertEqual(text.assembleFormattedText(A.underline['Hello, world.']), '\x1b[4mHello, world.')