import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_text_factory_confirm(self):
    with ui_testing.TestUIFactory('n\n') as ui:
        self.assertEqual(False, ui.confirm_action('Should %(thing)s pass?', 'breezy.tests.test_ui.confirmation', {'thing': 'this'}))