import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_select_mutables(document, comm):
    opts = {'A': [1, 2, 3], 'B': [2, 4, 6], 'C': dict(a=1, b=2)}
    select = Select(options=opts, value=opts['B'], name='Select')
    widget = select.get_root(document, comm=comm)
    assert isinstance(widget, select._widget_type)
    assert widget.title == 'Select'
    assert widget.value == str(opts['B'])
    assert widget.options == [(str(v), k) for k, v in opts.items()]
    widget.value = str(opts['B'])
    select._process_events({'value': str(opts['A'])})
    assert select.value == opts['A']
    widget.value = str(opts['B'])
    select._process_events({'value': str(opts['B'])})
    assert select.value == opts['B']
    select.value = opts['A']
    assert widget.value == str(opts['A'])