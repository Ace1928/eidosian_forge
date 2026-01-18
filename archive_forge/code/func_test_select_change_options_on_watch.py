import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_select_change_options_on_watch(document, comm):
    select = Select(options={'A': 'A', '1': 1, 'C': object}, value='A', name='Select')

    def set_options(event):
        if event.new == 1:
            select.options = {'D': 2, 'E': 'a'}
    select.param.watch(set_options, 'value')
    model = select.get_root(document, comm=comm)
    select.value = 1
    assert model.value == str(list(select.options.values())[0])
    assert model.options == [(str(v), k) for k, v in select.options.items()]