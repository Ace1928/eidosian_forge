import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_recursive_wrap():

    def dummy_model(name, layers):
        return Model(name, lambda model, X, is_train: ..., layers=layers)
    relu = Relu(5)
    chained = chain(relu, relu)
    chained_debug = wrap_model_recursive(chained, lambda model: dummy_model(f'dummy({model.name})', [model]))
    assert chained_debug.name == 'dummy(relu>>relu)'
    assert chained_debug.layers[0] is chained
    assert chained_debug.layers[0].layers[0].name == 'dummy(relu)'
    assert chained_debug.layers[0].layers[0].layers[0] is relu
    assert chained_debug.layers[0].layers[1].name == 'dummy(relu)'
    assert chained_debug.layers[0].layers[1].layers[0] is relu