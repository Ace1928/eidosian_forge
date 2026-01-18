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
@unittest.skipUnless(all(map(config.can_encode, 'å∂ßƒ')), 'Charset can not encode characters')
def test_show_source_not_formatted(self):
    self.repl.config.highlight_show_source = False
    self.repl.get_source_of_current_name = lambda: 'source code å∂ßƒåß∂ƒ'
    self.repl.show_source()