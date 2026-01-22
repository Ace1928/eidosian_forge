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
class ClearEnviron(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mock_environ = mock.patch.dict('os.environ', {'LC_ALL': os.environ.get('LC_ALL', 'C.UTF-8'), 'LANG': os.environ.get('LANG', 'C.UTF-8')}, clear=True)
        cls.mock_environ.start()
        TestCase.setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.mock_environ.stop()
        TestCase.tearDownClass()