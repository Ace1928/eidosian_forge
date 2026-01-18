import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_text_ui_choose_no_default(self):
    stdin_text = ' \n yes \nfoo\n'
    with ui_testing.TextUIFactory(stdin_text) as factory:
        self.assertEqual(0, factory.choose('', '&Yes\n&No'))
        self.assertEqual('foo\n', factory.stdin.read())