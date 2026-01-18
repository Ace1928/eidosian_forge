import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_select_change_groups(document, comm):
    groups = dict(A=dict(a=1, b=2), B=dict(c=3))
    select = Select(value=groups['A']['a'], groups=groups, name='Select')
    widget = select.get_root(document, comm=comm)
    new_groups = dict(C=dict(d=4), D=dict(e=5, f=6))
    select.groups = new_groups
    assert select.value == new_groups['C']['d']
    assert widget.value == str(new_groups['C']['d'])
    assert widget.options == {'C': [('4', 'd')], 'D': [('5', 'e'), ('6', 'f')]}
    select.groups = {}
    assert select.value is None
    assert widget.value == ''