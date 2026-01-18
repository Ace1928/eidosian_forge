from __future__ import annotations
import asyncio
import unittest.mock
from functools import partial
from typing import ClassVar, Mapping
import bokeh.core.properties as bp
import param
import pytest
from bokeh.document import Document
from bokeh.io.doc import patch_curdoc
from bokeh.models import Div
from panel.depends import bind, depends
from panel.layout import Tabs, WidgetBox
from panel.pane import Markdown
from panel.reactive import Reactive, ReactiveHTML
from panel.viewable import Viewable
from panel.widgets import (
def test_reactive_html_no_id_param_error():
    with pytest.raises(ValueError) as excinfo:

        class Test(ReactiveHTML):
            width = param.Number(default=200)
            _template = '<div width=${width}></div>'
    assert 'Found <div> node with the `width` attribute referencing the `width` parameter.' in str(excinfo.value)