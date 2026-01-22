from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class Norm(Operation):

    def __init__(self, ord=None, axis=None, keepdims=False):
        super().__init__()
        if isinstance(ord, str):
            if ord not in ('fro', 'nuc'):
                raise ValueError(f"Invalid `ord` argument. Expected one of {{'fro', 'nuc'}} when using string. Received: ord={ord}")
        if isinstance(axis, int):
            axis = [axis]
        self.ord = ord
        self.axis = axis
        self.keepdims = keepdims

    def compute_output_spec(self, x):
        output_dtype = backend.standardize_dtype(x.dtype)
        if 'int' in output_dtype or output_dtype == 'bool':
            output_dtype = backend.floatx()
        if self.axis is None:
            axis = tuple(range(len(x.shape)))
        else:
            axis = self.axis
        num_axes = len(axis)
        if num_axes == 1 and isinstance(self.ord, str):
            raise ValueError(f'Invalid `ord` argument for vector norm. Received: ord={self.ord}')
        elif num_axes == 2 and self.ord not in (None, 'fro', 'nuc', float('inf'), float('-inf'), 1, -1, 2, -2):
            raise ValueError(f'Invalid `ord` argument for matrix norm. Received: ord={self.ord}')
        return KerasTensor(reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims), dtype=output_dtype)

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return backend.linalg.norm(x, ord=self.ord, axis=self.axis, keepdims=self.keepdims)