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
def test_pass_parameterized_method_by_reference():

    class Test(param.Parameterized):
        a = param.Parameter(default=1)
        b = param.Parameter(default=2)

        @param.depends('a')
        def dep_a(self):
            return self.a

        @param.depends('dep_a', 'b')
        def dep_ab(self):
            return self.dep_a() + self.b
    test = Test()
    int_input = IntInput(start=0, end=400, value=test.dep_ab)
    assert int_input.value == 3
    test.a = 3
    assert int_input.value == 5
    test.b = 5
    assert int_input.value == 8