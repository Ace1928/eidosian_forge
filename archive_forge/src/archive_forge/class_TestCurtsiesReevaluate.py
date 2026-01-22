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
class TestCurtsiesReevaluate(TestCase):

    def setUp(self):
        self.repl = create_repl()

    def test_variable_is_cleared(self):
        self.repl._current_line = 'b = 10'
        self.repl.on_enter()
        self.assertIn('b', self.repl.interp.locals)
        self.repl.undo()
        self.assertNotIn('b', self.repl.interp.locals)