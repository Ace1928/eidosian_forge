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
class TestCompletionHelpers(TestCase):

    def test_gen_names(self):
        self.assertEqual(list(zip([1, 2, 3], gen_names())), [(1, 'a'), (2, 'b'), (3, 'c')])

    def test_completion_target(self):
        target = completion_target(14)
        self.assertEqual(len([x for x in dir(target) if not x.startswith('_')]), 14)