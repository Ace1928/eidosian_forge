import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_quietness(self):
    self.overrideEnv('BRZ_PROGRESS_BAR', 'text')
    ui_factory = ui_testing.TextUIFactory(stderr=ui_testing.StringIOAsTTY())
    with ui_factory:
        self.assertIsInstance(ui_factory._progress_view, _mod_ui_text.TextProgressView)
        ui_factory.be_quiet(True)
        self.assertIsInstance(ui_factory._progress_view, _mod_ui_text.NullProgressView)