import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
def test_set_state_transformer(echo):
    w = TransformerWidget()
    w.set_state(dict(d=[True, False, True]))
    expected = []
    if echo:
        expected.append(((), dict(buffers=[], data=dict(buffer_paths=[], method='echo_update', state=dict(d=[True, False, True])))))
    expected.append(((), dict(buffers=[], data=dict(buffer_paths=[], method='update', state=dict(d=[False, True, False])))))
    assert w.comm.messages == expected