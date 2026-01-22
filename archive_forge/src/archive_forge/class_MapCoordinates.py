from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import compute_conv_output_shape
class MapCoordinates(Operation):

    def __init__(self, order, fill_mode='constant', fill_value=0):
        super().__init__()
        self.order = order
        self.fill_mode = fill_mode
        self.fill_value = fill_value

    def call(self, image, coordinates):
        return backend.image.map_coordinates(image, coordinates, order=self.order, fill_mode=self.fill_mode, fill_value=self.fill_value)

    def compute_output_spec(self, image, coordinates):
        if coordinates.shape[0] != len(image.shape):
            raise ValueError(f'First dim of `coordinates` must be the same as the rank of `image`. Received image with shape: {image.shape} and coordinate leading dim of {coordinates.shape[0]}')
        if len(coordinates.shape) < 2:
            raise ValueError(f'Invalid coordinates rank: expected at least rank 2. Received input with shape: {coordinates.shape}')
        return KerasTensor(coordinates.shape[1:], dtype=image.dtype)