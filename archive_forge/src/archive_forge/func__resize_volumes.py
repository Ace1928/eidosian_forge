from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.utils import argument_validation
def _resize_volumes(self, x, depth_factor, height_factor, width_factor, data_format):
    """Resizes the volume contained in a 5D tensor.

        Args:
            x: Tensor or variable to resize.
            depth_factor: Positive integer.
            height_factor: Positive integer.
            width_factor: Positive integer.
            data_format: One of `"channels_first"`, `"channels_last"`.

        Returns:
            Resized tensor.
        """
    if data_format == 'channels_first':
        output = ops.repeat(x, depth_factor, axis=2)
        output = ops.repeat(output, height_factor, axis=3)
        output = ops.repeat(output, width_factor, axis=4)
        return output
    elif data_format == 'channels_last':
        output = ops.repeat(x, depth_factor, axis=1)
        output = ops.repeat(output, height_factor, axis=2)
        output = ops.repeat(output, width_factor, axis=3)
        return output
    else:
        raise ValueError(f'Invalid data_format: {data_format}')