import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_text_factory_unicode_password(self):
    """Test a unicode password."""
    ui = ui_testing.TextUIFactory('bazሴ')
    password = ui.get_password('Hello ሴ %(user)s', user='someሴ')
    self.assertEqual('bazሴ', password)
    self.assertEqual('Hello ሴ someሴ: ', ui.stderr.getvalue())
    self.assertEqual('', ui.stdin.readline())
    self.assertEqual('', ui.stdout.getvalue())