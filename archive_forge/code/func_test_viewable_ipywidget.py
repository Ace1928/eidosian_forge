import param
import pytest
from panel import config
from panel.interact import interactive
from panel.pane import Markdown, Str, panel
from panel.param import ParamMethod
from panel.viewable import Viewable, Viewer
from .util import jb_available
@jb_available
def test_viewable_ipywidget():
    pane = Str('A')
    with config.set(comms='ipywidgets'):
        data, metadata = pane._repr_mimebundle_()
    assert 'application/vnd.jupyter.widget-view+json' in data