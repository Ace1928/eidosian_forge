import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_replace_node():
    relu1 = Relu(5)
    relu2 = Relu(5)
    relu_chain = chain(relu1, relu2)
    relu1_debug = with_debug(relu1)
    debug = Model('test', lambda X: (X, lambda dY: dY), layers=[relu1, relu2, relu1, relu_chain], refs={'relu1': relu1, 'relu2': relu2, 'relu3': relu1})
    debug.replace_node(relu1, relu1_debug)
    assert debug.layers[0] == relu1_debug
    assert debug.layers[1] == relu2
    assert debug.layers[2] == relu1_debug
    assert debug.get_ref('relu1') == relu1_debug
    assert debug.get_ref('relu2') == relu2
    assert debug.get_ref('relu3') == relu1_debug
    assert debug.layers[3] == relu_chain
    assert debug.layers[3].layers[0] == relu1_debug
    assert debug.layers[3].layers[1] == relu2