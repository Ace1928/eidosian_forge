from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class ISTFT(Operation):

    def __init__(self, sequence_length, sequence_stride, fft_length, length=None, window='hann', center=True):
        super().__init__()
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.fft_length = fft_length
        self.length = length
        self.window = window
        self.center = center

    def compute_output_spec(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(f'Input `x` should be a tuple of two tensors - real and imaginary. Received: x={x}')
        real, imag = x
        if real.shape != imag.shape:
            raise ValueError(f'Input `x` should be a tuple of two tensors - real and imaginary. Both the real and imaginary parts should have the same shape. Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}')
        if len(real.shape) < 2:
            raise ValueError(f'Input should have rank >= 2. Received: input.shape = {real.shape}')
        if real.shape[-2] is not None:
            output_size = (real.shape[-2] - 1) * self.sequence_stride + self.fft_length
            if self.length is not None:
                output_size = self.length
            elif self.center:
                output_size = output_size - self.fft_length // 2 * 2
        else:
            output_size = None
        new_shape = real.shape[:-2] + (output_size,)
        return KerasTensor(shape=new_shape, dtype=real.dtype)

    def call(self, x):
        return backend.math.istft(x, sequence_length=self.sequence_length, sequence_stride=self.sequence_stride, fft_length=self.fft_length, length=self.length, window=self.window, center=self.center)