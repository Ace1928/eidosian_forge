from typing import Tuple, cast
import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import Model, NumpyOps, Softmax_v2
from thinc.types import Floats2d, Ints1d
from thinc.util import has_torch, torch2xp, xp2torch
def test_reject_incorrect_temperature():
    with pytest.raises(ValueError, match='softmax temperature.*zero'):
        Softmax_v2(normalize_outputs=False, temperature=0.0)
    model = Softmax_v2(normalize_outputs=False)
    model.attrs['softmax_temperature'] = 0.0
    model.initialize(inputs, outputs)
    with pytest.raises(ValueError, match='softmax temperature.*zero'):
        model(inputs, is_train=False)