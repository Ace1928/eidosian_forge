from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
def _solve_triangular(a, b, lower=False):
    a = backend.convert_to_tensor(a)
    b = backend.convert_to_tensor(b)
    _assert_2d(a)
    _assert_square(a)
    _assert_1d(b)
    _assert_a_b_compat(a, b)
    return backend.linalg.solve_triangular(a, b, lower)