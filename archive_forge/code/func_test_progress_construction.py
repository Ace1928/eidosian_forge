import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_progress_construction(self):
    """TextUIFactory constructs the right progress view.
        """
    FileStringIO = ui_testing.StringIOWithEncoding
    TTYStringIO = ui_testing.StringIOAsTTY
    for file_class, term, pb, expected_pb_class in ((TTYStringIO, 'xterm', 'none', _mod_ui_text.NullProgressView), (TTYStringIO, 'xterm', 'text', _mod_ui_text.TextProgressView), (TTYStringIO, 'xterm', None, _mod_ui_text.TextProgressView), (TTYStringIO, 'dumb', 'none', _mod_ui_text.NullProgressView), (TTYStringIO, 'dumb', 'text', _mod_ui_text.TextProgressView), (TTYStringIO, 'dumb', None, _mod_ui_text.NullProgressView), (FileStringIO, 'xterm', None, _mod_ui_text.NullProgressView), (FileStringIO, 'dumb', None, _mod_ui_text.NullProgressView), (FileStringIO, 'dumb', 'text', _mod_ui_text.TextProgressView)):
        self.overrideEnv('TERM', term)
        self.overrideEnv('BRZ_PROGRESS_BAR', pb)
        stdin = file_class('')
        stderr = file_class()
        stdout = file_class()
        uif = _mod_ui.make_ui_for_terminal(stdin, stdout, stderr)
        self.assertIsInstance(uif, _mod_ui_text.TextUIFactory, 'TERM={} BRZ_PROGRESS_BAR={} uif={!r}'.format(term, pb, uif))
        self.assertIsInstance(uif.make_progress_view(), expected_pb_class, 'TERM={} BRZ_PROGRESS_BAR={} uif={!r}'.format(term, pb, uif))