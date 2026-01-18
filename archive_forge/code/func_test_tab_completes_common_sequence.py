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
def test_tab_completes_common_sequence(self):
    self.repl._current_line = ' a'
    self.repl._cursor_offset = 2
    self.repl.matches_iter.matches = ['aaa', 'aab', 'aac']
    self.repl.matches_iter.is_cseq.return_value = True
    self.repl.matches_iter.substitute_cseq.return_value = (None, None)
    self.repl.on_tab()
    self.repl.matches_iter.substitute_cseq.assert_called_once_with()