import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
@pytest.mark.parametrize('op_name', ['abs', 'exp', 'sqrt', 'tanh'])
def test_unary_arithmetic(op_name):
    numpy_flat_arr = numpy.random.randint(-100, 100, size=100)
    modin_flat_arr = np.array(numpy_flat_arr)
    assert_scalar_or_array_equal(getattr(np, op_name)(modin_flat_arr), getattr(numpy, op_name)(numpy_flat_arr))
    numpy_arr = numpy_flat_arr.reshape((10, 10))
    modin_arr = np.array(numpy_arr)
    assert_scalar_or_array_equal(getattr(np, op_name)(modin_arr), getattr(numpy, op_name)(numpy_arr))