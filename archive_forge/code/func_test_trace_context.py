import breezy
from .. import library_state, tests
from .. import ui as _mod_ui
from . import fixtures
def test_trace_context(self):
    self.overrideAttr(breezy, '_global_state', None)
    tracer = fixtures.RecordingContextManager()
    ui = _mod_ui.SilentUIFactory()
    state = library_state.BzrLibraryState(ui=ui, trace=tracer)
    state.__enter__()
    try:
        self.assertEqual(['__enter__'], tracer._calls)
    finally:
        state.__exit__(None, None, None)
        self.assertEqual(['__enter__', '__exit__'], tracer._calls)