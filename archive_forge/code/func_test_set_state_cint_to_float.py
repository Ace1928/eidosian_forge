import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
def test_set_state_cint_to_float(echo):
    w = NumberWidget()
    w.set_state(dict(ci=5.6))
    assert len(w.comm.messages) == (2 if echo else 1)
    msg = w.comm.messages[-1]
    data = msg[1]['data']
    assert data['method'] == 'update'
    assert data['state'] == {'ci': 5}