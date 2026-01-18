import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_text_tick_after_update(self):
    ui_factory = ui_testing.TextUIFactory()
    with ui_factory.nested_progress_bar() as pb:
        pb.update('task', 0, 3)
        ui_factory._progress_view._last_repaint = time.time() - 1.0
        pb.tick()