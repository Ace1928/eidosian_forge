import code
import os
import sys
import tempfile
import io
from typing import cast
import unittest
from contextlib import contextmanager
from functools import partial
from unittest import mock
from bpython.curtsiesfrontend import repl as curtsiesrepl
from bpython.curtsiesfrontend import interpreter
from bpython.curtsiesfrontend import events as bpythonevents
from bpython.curtsiesfrontend.repl import LineType
from bpython import autocomplete
from bpython import config
from bpython import args
from bpython.test import (
from curtsies import events
from curtsies.window import CursorAwareWindow
from importlib import invalidate_caches
def test_tab_after_indentation_adds_space(self):
    self.repl._current_line = '    '
    self.repl._cursor_offset = 4
    self.repl.on_tab()
    self.assertEqual(self.repl._current_line, '        ')
    self.assertEqual(self.repl._cursor_offset, 8)