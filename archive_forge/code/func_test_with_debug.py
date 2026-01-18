import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_with_debug():
    pytest.importorskip('ml_datasets')
    import ml_datasets
    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
    counts = Counter()

    def on_init(*_):
        counts['init'] += 1

    def on_forward(*_):
        counts['forward'] += 1

    def on_backprop(*_):
        counts['backprop'] += 1
    relu = Relu()
    relu2 = with_debug(Relu(), on_init=on_init, on_forward=on_forward, on_backprop=on_backprop)
    chained = chain(relu, relu2, relu2)
    chained.initialize(X=train_X[:5], Y=train_Y[:5])
    _, backprop = chained(X=train_X[:5], is_train=False)
    backprop(train_Y[:5])
    assert counts == {'init': 2, 'forward': 4, 'backprop': 2}