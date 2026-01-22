from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class IRFFT(Operation):

    def __init__(self, fft_length=None):
        super().__init__()
        self.fft_length = fft_length

    def compute_output_spec(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(f'Input `x` should be a tuple of two tensors - real and imaginary. Received: x={x}')
        real, imag = x
        if real.shape != imag.shape:
            raise ValueError(f'Input `x` should be a tuple of two tensors - real and imaginary. Both the real and imaginary parts should have the same shape. Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}')
        if len(real.shape) < 1:
            raise ValueError(f'Input should have rank >= 1. Received: input.shape = {real.shape}')
        if self.fft_length is not None:
            new_last_dimension = self.fft_length
        elif real.shape[-1] is not None:
            new_last_dimension = 2 * (real.shape[-1] - 1)
        else:
            new_last_dimension = None
        new_shape = real.shape[:-1] + (new_last_dimension,)
        return KerasTensor(shape=new_shape, dtype=real.dtype)

    def call(self, x):
        return backend.math.irfft(x, fft_length=self.fft_length)