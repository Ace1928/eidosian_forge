import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
def test_set_state_float_to_float(echo):
    w = NumberWidget()
    w.set_state(dict(f=1.2, cf=2.6))
    assert len(w.comm.messages) == (1 if echo else 0)