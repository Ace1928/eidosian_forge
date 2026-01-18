import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_select_non_hashable_options(document, comm):
    opts = {'A': np.array([1, 2, 3]), '1': np.array([3, 4, 5])}
    select = Select(options=opts, value=opts['1'], name='Select')
    widget = select.get_root(document, comm=comm)
    select.value = opts['A']
    assert select.value is opts['A']
    assert widget.value == str(opts['A'])
    opts.pop('A')
    select.options = opts
    assert select.value is opts['1']
    assert widget.value == str(opts['1'])