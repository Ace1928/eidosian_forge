import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
@pytest.mark.parametrize('options', [[10, 20], dict(A=10, B=20)], ids=['list', 'dict'])
@pytest.mark.parametrize('size', [1, 2], ids=['size=1', 'size>1'])
def test_select_disabled_options_after_init(options, size, document, comm):
    select = Select(options=options, size=size)
    select.disabled_options = [20]
    widget = select.get_root(document, comm=comm)
    assert isinstance(widget, select._widget_type)
    assert widget.disabled_options == [20]