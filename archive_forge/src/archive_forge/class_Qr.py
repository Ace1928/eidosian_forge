from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class Qr(Operation):

    def __init__(self, mode='reduced'):
        super().__init__()
        if mode not in {'reduced', 'complete'}:
            raise ValueError(f"`mode` argument value not supported. Expected one of {{'reduced', 'complete'}}. Received: mode={mode}")
        self.mode = mode

    def compute_output_spec(self, x):
        if len(x.shape) < 2:
            raise ValueError(f'Input should have rank >= 2. Received: input.shape = {x.shape}')
        m = x.shape[-2]
        n = x.shape[-1]
        if m is None or n is None:
            raise ValueError(f'Input should have its last 2 dimensions fully-defined. Received: input.shape = {x.shape}')
        k = min(m, n)
        base = tuple(x.shape[:-2])
        if self.mode == 'reduced':
            return (KerasTensor(shape=base + (m, k), dtype=x.dtype), KerasTensor(shape=base + (k, n), dtype=x.dtype))
        return (KerasTensor(shape=base + (m, m), dtype=x.dtype), KerasTensor(shape=base + (m, n), dtype=x.dtype))

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return backend.linalg.qr(x, mode=self.mode)