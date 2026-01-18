from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_single_value_int():
    for a in (1, 5, -3, 0):
        if not a:
            expected_min = 0
            expected_max = 1
        elif a > 0:
            expected_min = -a
            expected_max = 3 * a
        else:
            expected_min = 3 * a
            expected_max = -a
        c = interactive(f, a=a)
        assert len(c.children) == 2
        w = c.children[0]
        check_widget(w, cls=widgets.IntSlider, description='a', value=a, min=expected_min, max=expected_max, step=1, readout=True)