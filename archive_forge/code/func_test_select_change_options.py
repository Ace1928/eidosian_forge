import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_select_change_options(document, comm):
    opts = {'A': 'a', '1': 1}
    select = Select(options=opts, value=opts['1'], name='Select')
    widget = select.get_root(document, comm=comm)
    select.options = {'A': 'a'}
    assert select.value == opts['A']
    assert widget.value == str(opts['A'])
    select.options = {}
    assert select.value is None
    assert widget.value == ''