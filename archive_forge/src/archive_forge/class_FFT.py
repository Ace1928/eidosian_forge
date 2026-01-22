from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class FFT(Operation):

    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def compute_output_spec(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(f'Input `x` should be a tuple of two tensors - real and imaginary. Received: x={x}')
        real, imag = x
        if real.shape != imag.shape:
            raise ValueError(f'Input `x` should be a tuple of two tensors - real and imaginary. Both the real and imaginary parts should have the same shape. Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}')
        if len(real.shape) < 1:
            raise ValueError(f'Input should have rank >= 1. Received: input.shape = {real.shape}')
        m = real.shape[-1]
        if m is None:
            raise ValueError(f'Input should have its {self.axis}th axis fully-defined. Received: input.shape = {real.shape}')
        return (KerasTensor(shape=real.shape, dtype=real.dtype), KerasTensor(shape=imag.shape, dtype=imag.dtype))

    def call(self, x):
        return backend.math.fft(x)