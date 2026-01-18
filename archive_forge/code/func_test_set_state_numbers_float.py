import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
def test_set_state_numbers_float(echo):
    w = NumberWidget()
    w.set_state(dict(f=1.0, cf=2.0, ci=4.0))
    assert len(w.comm.messages) == (1 if echo else 0)