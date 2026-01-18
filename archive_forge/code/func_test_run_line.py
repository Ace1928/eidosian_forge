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
def test_run_line(self):
    try:
        orig_stdout = sys.stdout
        sys.stdout = self.repl.stdout
        [self.repl.add_normal_character(c) for c in '1 + 1']
        self.repl.on_enter(new_code=False)
        screen = fsarray(['>>> 1 + 1', '2', 'Welcome to'])
        self.assert_paint_ignoring_formatting(screen, (1, 1))
    finally:
        sys.stdout = orig_stdout