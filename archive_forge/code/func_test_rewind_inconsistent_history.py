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
def test_rewind_inconsistent_history(self):
    self.enter('1 + 1')
    self.enter('2 + 2')
    self.enter('3 + 3')
    screen = ['>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> 3 + 3', '6', '>>> ']
    self.assert_paint_ignoring_formatting(screen, (6, 4))
    self.repl.scroll_offset += len(screen) - self.repl.height
    self.assert_paint_ignoring_formatting(screen[2:], (4, 4))
    self.repl.display_lines[0] = self.repl.display_lines[0] * 2
    self.undo()
    screen = [INCONSISTENT_HISTORY_MSG[:self.repl.width], '>>> 2 + 2', '4', '>>> ', '', ' ']
    self.assert_paint_ignoring_formatting(screen, (3, 4))
    self.repl.scroll_offset += len(screen) - self.repl.height
    self.assert_paint_ignoring_formatting(screen[1:-2], (2, 4))
    self.assert_paint_ignoring_formatting(screen[1:-2], (2, 4))