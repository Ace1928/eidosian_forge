from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
class CharacterAttributeTests(unittest.TestCase):
    """
    Tests for L{twisted.conch.insults.helper.CharacterAttribute}.
    """

    def test_equality(self) -> None:
        """
        L{CharacterAttribute}s must have matching character attribute values
        (bold, blink, underline, etc) with the same values to be considered
        equal.
        """
        self.assertEqual(helper.CharacterAttribute(), helper.CharacterAttribute())
        self.assertEqual(helper.CharacterAttribute(), helper.CharacterAttribute(charset=G0))
        self.assertEqual(helper.CharacterAttribute(bold=True, underline=True, blink=False, reverseVideo=True, foreground=helper.BLUE), helper.CharacterAttribute(bold=True, underline=True, blink=False, reverseVideo=True, foreground=helper.BLUE))
        self.assertNotEqual(helper.CharacterAttribute(), helper.CharacterAttribute(charset=G1))
        self.assertNotEqual(helper.CharacterAttribute(bold=True), helper.CharacterAttribute(bold=False))

    def test_wantOneDeprecated(self) -> None:
        """
        L{twisted.conch.insults.helper.CharacterAttribute.wantOne} emits
        a deprecation warning when invoked.
        """
        helper._FormattingState().wantOne(bold=True)
        warningsShown = self.flushWarnings([self.test_wantOneDeprecated])
        self.assertEqual(len(warningsShown), 1)
        self.assertEqual(warningsShown[0]['category'], DeprecationWarning)
        deprecatedClass = 'twisted.conch.insults.helper._FormattingState.wantOne'
        self.assertEqual(warningsShown[0]['message'], '%s was deprecated in Twisted 13.1.0' % deprecatedClass)