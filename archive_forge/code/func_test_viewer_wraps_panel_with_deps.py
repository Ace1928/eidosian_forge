import param
import pytest
from panel import config
from panel.interact import interactive
from panel.pane import Markdown, Str, panel
from panel.param import ParamMethod
from panel.viewable import Viewable, Viewer
from .util import jb_available
def test_viewer_wraps_panel_with_deps(document, comm):
    tv = ExampleViewerWithDeps(value='hello')
    view = tv._create_view()
    view.get_root(document, comm)
    assert isinstance(view, ParamMethod)
    assert view._pane.object == 'hello'
    tv.value = 'goodbye'
    assert view._pane.object == 'goodbye'