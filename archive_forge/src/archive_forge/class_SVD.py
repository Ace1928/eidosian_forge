from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class SVD(Operation):

    def __init__(self, full_matrices=True, compute_uv=True):
        super().__init__()
        self.full_matrices = full_matrices
        self.compute_uv = compute_uv

    def call(self, x):
        return _svd(x, self.full_matrices, self.compute_uv)

    def compute_output_spec(self, x):
        _assert_2d(x)
        rows, columns = x.shape[-2:]
        batches = x.shape[:-2]
        s_shape = batches + (min(rows, columns),)
        if self.full_matrices:
            u_shape = batches + (rows, rows)
            v_shape = batches + (columns, columns)
        else:
            u_shape = batches + (rows, min(rows, columns))
            v_shape = batches + (min(rows, columns), columns)
        if self.compute_uv:
            return (KerasTensor(u_shape, x.dtype), KerasTensor(s_shape, x.dtype), KerasTensor(v_shape, x.dtype))
        return KerasTensor(s_shape, x.dtype)