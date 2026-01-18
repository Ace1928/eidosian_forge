import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
@pytest.mark.parametrize('axis', [0, 1])
def test_split_2d(axis):
    x = numpy.random.randint(-100, 100, size=(6, 4))
    numpy_result = numpy.split(x, 2, axis=axis)
    modin_result = np.split(np.array(x), 2, axis=axis)
    for modin_entry, numpy_entry in zip(modin_result, numpy_result):
        assert_scalar_or_array_equal(modin_entry, numpy_entry)
    idxs = [2, 3]
    numpy_result = numpy.split(x, idxs, axis=axis)
    modin_result = np.split(np.array(x), idxs, axis=axis)
    for modin_entry, numpy_entry in zip(modin_result, numpy_result):
        assert_scalar_or_array_equal(modin_entry, numpy_entry)