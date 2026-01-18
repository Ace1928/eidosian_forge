import breezy
from .. import library_state, tests
from .. import ui as _mod_ui
from . import fixtures
def test_ui_not_specified(self):
    self.overrideAttr(breezy, '_global_state', None)
    state = library_state.BzrLibraryState(ui=None, trace=fixtures.RecordingContextManager())
    orig_ui = _mod_ui.ui_factory
    state.__enter__()
    try:
        self.assertEqual(orig_ui, _mod_ui.ui_factory)
    finally:
        state.__exit__(None, None, None)
        self.assertEqual(orig_ui, _mod_ui.ui_factory)