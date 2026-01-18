import param
import pytest
from panel.io import block_comm
from panel.layout import Row
from panel.links import CallbackGenerator
from panel.tests.util import check_layoutable_properties
from panel.util import param_watchers
from panel.widgets import (
from panel.widgets.tables import BaseTable
def test_widget_from_param_instance():

    class Test(param.Parameterized):
        a = param.Parameter()
    test = Test()
    widget = TextInput.from_param(test.param.a)
    assert isinstance(widget, TextInput)
    assert widget.name == 'A'
    test.a = 'abc'
    assert widget.value == 'abc'
    widget.value = 'def'
    assert test.a == 'def'