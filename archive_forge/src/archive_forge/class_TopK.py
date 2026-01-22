from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class TopK(Operation):

    def __init__(self, k, sorted=False):
        super().__init__()
        self.k = k
        self.sorted = sorted

    def compute_output_spec(self, x):
        output_shape = list(x.shape)
        output_shape[-1] = self.k
        return (KerasTensor(shape=output_shape, dtype=x.dtype), KerasTensor(shape=output_shape, dtype='int32'))

    def call(self, x):
        return backend.math.top_k(x, self.k, self.sorted)