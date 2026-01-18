import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
def test_split_2d_oob():
    x = numpy.random.randint(-100, 100, size=(6, 4))
    idxs = [2, 3, 6]
    numpy_result = numpy.split(x, idxs)
    modin_result = np.split(np.array(x), idxs)
    for modin_entry, numpy_entry in zip(modin_result, numpy_result):
        assert_scalar_or_array_equal(modin_entry, numpy_entry)