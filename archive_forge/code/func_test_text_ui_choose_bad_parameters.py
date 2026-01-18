import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_text_ui_choose_bad_parameters(self):
    with ui_testing.TextUIFactory('') as factory:
        self.assertRaises(ValueError, factory.choose, '', '&Yes\n&No', 3)
        self.assertRaises(ValueError, factory.choose, '', '&choice\n&ChOiCe')
        self.assertRaises(ValueError, factory.choose, '', '&choice1\nchoi&ce2')