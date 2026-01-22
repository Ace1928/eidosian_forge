from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
class RedirectWithNoLocationTests(unittest.TestCase):
    """
    L{RedirectWithNoLocation} is a subclass of L{Error} which sets
    a custom message in the constructor.
    """

    def test_validMessage(self) -> None:
        """
        When C{code}, C{message}, and C{uri} are passed to the
        L{RedirectWithNoLocation} constructor, the C{message} and C{uri}
        attributes are set, respectively.
        """
        e = error.RedirectWithNoLocation(b'302', b'REDIRECT', b'https://example.com')
        self.assertEqual(e.message, b'REDIRECT to https://example.com')
        self.assertEqual(e.uri, b'https://example.com')