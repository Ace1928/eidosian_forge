from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
def test_validCode(self) -> None:
    m = error._codeToMessage(b'302')
    self.assertEqual(m, b'Found')