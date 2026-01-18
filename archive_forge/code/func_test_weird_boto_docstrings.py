import itertools
import os
import pydoc
import string
import sys
from contextlib import contextmanager
from typing import cast
from curtsies.formatstringarray import (
from curtsies.fmtfuncs import cyan, bold, green, yellow, on_magenta, red
from curtsies.window import CursorAwareWindow
from unittest import mock, skipIf
from bpython.curtsiesfrontend.events import RefreshRequestEvent
from bpython import config, inspection
from bpython.curtsiesfrontend.repl import BaseRepl
from bpython.curtsiesfrontend import replpainter
from bpython.curtsiesfrontend.repl import (
from bpython.test import FixLanguageTestCase as TestCase, TEST_CONFIG
def test_weird_boto_docstrings(self):

    class WeirdDocstring(str):

        def expandtabs(self, tabsize=8):
            return 'asdfåß∂ƒ'.expandtabs(tabsize)

    def foo():
        pass
    foo.__doc__ = WeirdDocstring()
    wd = pydoc.getdoc(foo)
    actual = replpainter.formatted_docstring(wd, 40, config=setup_config())
    expected = fsarray(['asdfåß∂ƒ'])
    assertFSArraysEqualIgnoringFormatting(actual, expected)