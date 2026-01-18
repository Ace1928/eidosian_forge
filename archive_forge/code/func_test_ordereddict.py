from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_ordereddict():
    from collections import OrderedDict
    items = [(3, 300), (1, 100), (2, 200)]
    first = items[0][1]
    values = OrderedDict(items)
    c = interactive(f, lis=values)
    assert len(c.children) == 2
    d = dict(cls=widgets.Dropdown, value=first, options=values, _options_labels=('3', '1', '2'), _options_values=(300, 100, 200))
    check_widget_children(c, lis=d)