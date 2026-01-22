from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
class CodeToMessageTests(unittest.TestCase):
    """
    L{_codeToMessages} inverts L{_responses.RESPONSES}
    """

    def test_validCode(self) -> None:
        m = error._codeToMessage(b'302')
        self.assertEqual(m, b'Found')

    def test_invalidCode(self) -> None:
        m = error._codeToMessage(b'987')
        self.assertEqual(m, None)

    def test_nonintegerCode(self) -> None:
        m = error._codeToMessage(b'InvalidCode')
        self.assertEqual(m, None)