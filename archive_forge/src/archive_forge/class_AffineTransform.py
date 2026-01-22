from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import compute_conv_output_shape
class AffineTransform(Operation):

    def __init__(self, interpolation='bilinear', fill_mode='constant', fill_value=0, data_format='channels_last'):
        super().__init__()
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.data_format = data_format

    def call(self, image, transform):
        return backend.image.affine_transform(image, transform, interpolation=self.interpolation, fill_mode=self.fill_mode, fill_value=self.fill_value, data_format=self.data_format)

    def compute_output_spec(self, image, transform):
        if len(image.shape) not in (3, 4):
            raise ValueError(f'Invalid image rank: expected rank 3 (single image) or rank 4 (batch of images). Received input with shape: image.shape={image.shape}')
        if len(transform.shape) not in (1, 2):
            raise ValueError(f'Invalid transform rank: expected rank 1 (single transform) or rank 2 (batch of transforms). Received input with shape: transform.shape={transform.shape}')
        return KerasTensor(image.shape, dtype=image.dtype)