from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_single_value_string():
    a = 'hello'
    c = interactive(f, a=a)
    w = c.children[0]
    check_widget(w, cls=widgets.Text, description='a', value=a)