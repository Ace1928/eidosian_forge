from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import compute_conv_output_shape
def _pad_images(images, top_padding, bottom_padding, left_padding, right_padding, target_height, target_width):
    images = backend.convert_to_tensor(images)
    is_batch = True
    images_shape = ops.shape(images)
    if len(images_shape) == 3:
        is_batch = False
        images = backend.numpy.expand_dims(images, 0)
    elif len(images_shape) != 4:
        raise ValueError(f'Invalid shape for argument `images`: it must have rank 3 or 4. Received: images.shape={images_shape}')
    batch, height, width, depth = ops.shape(images)
    if [top_padding, bottom_padding, target_height].count(None) != 1:
        raise ValueError(f'Must specify exactly two of top_padding, bottom_padding, target_height. Received: top_padding={top_padding}, bottom_padding={bottom_padding}, target_height={target_height}')
    if [left_padding, right_padding, target_width].count(None) != 1:
        raise ValueError(f'Must specify exactly two of left_padding, right_padding, target_width. Received: left_padding={left_padding}, right_padding={right_padding}, target_width={target_width}')
    if top_padding is None:
        top_padding = target_height - bottom_padding - height
    if bottom_padding is None:
        bottom_padding = target_height - top_padding - height
    if left_padding is None:
        left_padding = target_width - right_padding - width
    if right_padding is None:
        right_padding = target_width - left_padding - width
    if top_padding < 0:
        raise ValueError(f'top_padding must be >= 0. Received: top_padding={top_padding}')
    if left_padding < 0:
        raise ValueError(f'left_padding must be >= 0. Received: left_padding={left_padding}')
    if right_padding < 0:
        raise ValueError(f'right_padding must be >= 0. Received: right_padding={right_padding}')
    if bottom_padding < 0:
        raise ValueError(f'bottom_padding must be >= 0. Received: bottom_padding={bottom_padding}')
    paddings = backend.numpy.reshape(backend.numpy.stack([0, 0, top_padding, bottom_padding, left_padding, right_padding, 0, 0]), [4, 2])
    padded = backend.numpy.pad(images, paddings)
    if target_height is None:
        target_height = top_padding + height + bottom_padding
    if target_width is None:
        target_width = left_padding + width + right_padding
    padded_shape = [batch, target_height, target_width, depth]
    padded = backend.numpy.reshape(padded, padded_shape)
    if not is_batch:
        padded = backend.numpy.squeeze(padded, axis=[0])
    return padded