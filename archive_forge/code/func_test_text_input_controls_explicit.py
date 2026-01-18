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
def test_text_input_controls_explicit():
    text_input = TextInput()
    controls = text_input.controls(['placeholder', 'disabled'])
    assert isinstance(controls, WidgetBox)
    assert len(controls) == 3
    name, disabled, placeholder = controls
    assert isinstance(name, StaticText)
    assert isinstance(disabled, Checkbox)
    assert isinstance(placeholder, TextInput)
    text_input.disabled = True
    assert disabled.value
    text_input.placeholder = 'Test placeholder...'
    assert placeholder.value == 'Test placeholder...'