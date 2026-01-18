import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_text_factory_ascii_password(self):
    ui = ui_testing.TestUIFactory('secret\n')
    with ui.nested_progress_bar():
        self.assertEqual('secret', self.apply_redirected(ui.stdin, ui.stdout, ui.stderr, ui.get_password))
        self.assertEqual(': ', ui.stderr.getvalue())
        self.assertEqual('', ui.stdout.readline())
        self.assertEqual('', ui.stdin.readline())