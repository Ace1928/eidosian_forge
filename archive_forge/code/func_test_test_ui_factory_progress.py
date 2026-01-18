import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_test_ui_factory_progress(self):
    ui = ui_testing.TestUIFactory()
    with ui.nested_progress_bar() as pb:
        pb.update('hello')
        pb.tick()