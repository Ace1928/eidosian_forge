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
def test_param_rename():
    """Test that Reactive renames params and properties"""

    class ReactiveRename(Reactive):
        a = param.Parameter()
        _rename: ClassVar[Mapping[str, str | None]] = {'a': 'b'}
    obj = ReactiveRename()
    params = obj._process_property_change({'b': 1})
    assert params == {'a': 1}
    properties = obj._process_param_change({'a': 1})
    assert properties == {'b': 1}