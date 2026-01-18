import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_text_ui_get_integer(self):
    stdin_text = '1\n  -2  \nhmmm\nwhat else ?\nCome on\nok 42\n4.24\n42\n'
    with ui_testing.TextUIFactory(stdin_text) as factory:
        self.assertEqual(1, factory.get_integer(''))
        self.assertEqual(-2, factory.get_integer(''))
        self.assertEqual(42, factory.get_integer(''))