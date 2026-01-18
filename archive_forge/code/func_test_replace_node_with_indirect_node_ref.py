import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_replace_node_with_indirect_node_ref():

    def dummy_model(name, layers):
        return Model(name, lambda model, X, is_train: ..., layers=layers)
    y = dummy_model('y', [])
    x = dummy_model('x', [y])
    y_debug = with_debug(y)
    b = dummy_model('b', [x])
    b.set_ref('y', y)
    a = chain(x, b)
    a.name = 'a'
    a.replace_node(y, y_debug)
    assert a.layers[0].layers[0] == y_debug
    assert a.layers[1].layers[0].layers[0] == y_debug
    assert a.layers[1].get_ref('y') == y_debug