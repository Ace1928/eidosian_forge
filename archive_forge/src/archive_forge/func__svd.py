from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
def _svd(x, full_matrices=True, compute_uv=True):
    x = backend.convert_to_tensor(x)
    _assert_2d(x)
    return backend.linalg.svd(x, full_matrices, compute_uv)