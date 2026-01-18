import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
def test_set_state_simple(echo):
    w = SimpleWidget()
    w.set_state(dict(a=True, b=[True, False, True], c=[False, True, False]))
    assert len(w.comm.messages) == (1 if echo else 0)