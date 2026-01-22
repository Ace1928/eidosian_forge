from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import compute_conv_output_shape
def _extract_patches(image, size, strides=None, dilation_rate=1, padding='valid', data_format='channels_last'):
    if isinstance(size, int):
        patch_h = patch_w = size
    elif len(size) == 2:
        patch_h, patch_w = (size[0], size[1])
    else:
        raise TypeError(f'Invalid `size` argument. Expected an int or a tuple of length 2. Received: size={size}')
    if data_format == 'channels_last':
        channels_in = image.shape[-1]
    elif data_format == 'channels_first':
        channels_in = image.shape[-3]
    if not strides:
        strides = size
    out_dim = patch_h * patch_w * channels_in
    kernel = backend.numpy.eye(out_dim)
    kernel = backend.numpy.reshape(kernel, (patch_h, patch_w, channels_in, out_dim))
    _unbatched = False
    if len(image.shape) == 3:
        _unbatched = True
        image = backend.numpy.expand_dims(image, axis=0)
    patches = backend.nn.conv(inputs=image, kernel=kernel, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate)
    if _unbatched:
        patches = backend.numpy.squeeze(patches, axis=0)
    return patches