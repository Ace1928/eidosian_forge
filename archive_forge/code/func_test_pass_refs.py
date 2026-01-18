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
def test_pass_refs():
    slider = IntSlider(value=5, start=1, end=10, name='Number')
    size = IntSlider(value=12, start=6, end=24, name='Size')

    def refs(number, size):
        return {'object': '*' * number, 'styles': {'font-size': f'{size}pt'}}
    irefs = bind(refs, slider, size)
    md = Markdown(refs=irefs)
    assert md.object == '*****'
    assert md.styles == {'font-size': '12pt'}
    slider.value = 3
    assert md.object == '***'
    size.value = 7
    assert md.styles == {'font-size': '7pt'}