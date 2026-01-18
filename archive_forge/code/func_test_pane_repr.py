import param
import pytest
import panel as pn
from panel.chat import ChatMessage
from panel.config import config
from panel.interact import interactive
from panel.io.loading import LOADING_INDICATOR_CSS_CLASS
from panel.layout import Row
from panel.links import CallbackGenerator
from panel.pane import (
from panel.param import (
from panel.tests.util import check_layoutable_properties
from panel.util import param_watchers
def test_pane_repr(document, comm):
    pane = pn.panel('Some text', width=400)
    assert repr(pane) == 'Markdown(str, width=400)'