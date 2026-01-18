import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_text_ui_getusername(self):
    ui = ui_testing.TextUIFactory('someuser\n\n')
    self.assertEqual('someuser', ui.get_username('Hello %(host)s', host='some'))
    self.assertEqual('Hello some: ', ui.stderr.getvalue())
    self.assertEqual('', ui.stdout.getvalue())
    self.assertEqual('', ui.get_username('Gebruiker'))
    self.assertEqual('', ui.stdin.readline())