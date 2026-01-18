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
def test_rewind_history_not_quite_inconsistent(self):
    self.repl.width = 50
    sys.a = 5
    self.enter("for i in range(__import__('sys').a):")
    self.enter('    print(i)')
    self.enter('')
    self.enter('1 + 1')
    self.enter('2 + 2')
    screen = [">>> for i in range(__import__('sys').a):", '...     print(i)', '... ', '0', '1', '2', '3', '4', '>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> ']
    self.assert_paint_ignoring_formatting(screen, (12, 4))
    self.repl.scroll_offset += len(screen) - self.repl.height
    self.assert_paint_ignoring_formatting(screen[8:], (4, 4))
    sys.a = 6
    self.undo()
    screen = ['5', '>>> 1 + 1', '2', '>>> ']
    self.assert_paint_ignoring_formatting(screen, (3, 4))