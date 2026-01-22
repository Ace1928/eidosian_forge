from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import compute_conv_output_shape
class PadImages(Operation):

    def __init__(self, top_padding, bottom_padding, left_padding, right_padding, target_height, target_width):
        super().__init__()
        self.top_padding = top_padding
        self.bottom_padding = bottom_padding
        self.left_padding = left_padding
        self.right_padding = right_padding
        self.target_height = target_height
        self.target_width = target_width

    def call(self, images):
        return _pad_images(images, self.top_padding, self.bottom_padding, self.left_padding, self.right_padding, self.target_height, self.target_width)

    def compute_output_spec(self, images):
        images_shape = ops.shape(images)
        if self.target_height is None:
            height_axis = 0 if len(images_shape) == 3 else 1
            self.target_height = self.top_padding + images_shape[height_axis] + self.bottom_padding
        if self.target_width is None:
            width_axis = 0 if len(images_shape) == 3 else 2
            self.target_width = self.left_padding + images_shape[width_axis] + self.right_padding
        out_shape = (images_shape[0], self.target_height, self.target_width, images_shape[-1])
        if len(images_shape) == 3:
            out_shape = out_shape[1:]
        return KerasTensor(shape=out_shape, dtype=images.dtype)