import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_silent_ui_getbool(self):
    factory = _mod_ui.SilentUIFactory()
    stdout = ui_testing.StringIOWithEncoding()
    self.assertRaises(NotImplementedError, self.apply_redirected, None, stdout, stdout, factory.get_boolean, 'foo')