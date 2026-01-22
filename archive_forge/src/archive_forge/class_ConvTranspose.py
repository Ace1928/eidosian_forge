from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend import standardize_data_format
from keras.src.backend.common.backend_utils import (
from keras.src.ops import operation_utils
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class ConvTranspose(Operation):

    def __init__(self, strides, padding='valid', output_padding=None, data_format=None, dilation_rate=1):
        super().__init__()
        self.strides = strides
        self.output_padding = output_padding
        self.padding = padding.lower()
        self.data_format = data_format
        self.dilation_rate = dilation_rate

    def call(self, inputs, kernel):
        return backend.nn.conv_transpose(inputs, kernel, self.strides, self.output_padding, self.padding, self.data_format, self.dilation_rate)

    def compute_output_spec(self, inputs, kernel):
        kernel_size = kernel.shape[:-2]
        filters = kernel.shape[-2]
        output_shape = compute_conv_transpose_output_shape(inputs.shape, kernel_size, filters, self.strides, self.padding, self.output_padding, self.data_format, self.dilation_rate)
        return KerasTensor(output_shape, dtype=inputs.dtype)