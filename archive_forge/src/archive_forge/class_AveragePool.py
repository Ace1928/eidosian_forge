from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend import standardize_data_format
from keras.src.backend.common.backend_utils import (
from keras.src.ops import operation_utils
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class AveragePool(Operation):

    def __init__(self, pool_size, strides=None, padding='valid', data_format=None):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding.lower()
        self.data_format = data_format

    def call(self, inputs):
        return backend.nn.average_pool(inputs, self.pool_size, self.strides, self.padding, self.data_format)

    def compute_output_spec(self, inputs):
        output_shape = operation_utils.compute_pooling_output_shape(inputs.shape, self.pool_size, self.strides, self.padding, self.data_format)
        return KerasTensor(output_shape, dtype=inputs.dtype)