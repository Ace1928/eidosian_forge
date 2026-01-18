import functools
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('image.yuv_to_rgb')
@dispatch.add_dispatch_support
def yuv_to_rgb(images):
    """Converts one or more images from YUV to RGB.

  Outputs a tensor of the same shape as the `images` tensor, containing the RGB
  value of the pixels.
  The output is only well defined if the Y value in images are in [0,1],
  U and V value are in [-0.5,0.5].

  As per the above description, you need to scale your YUV images if their
  pixel values are not in the required range. Below given example illustrates
  preprocessing of each channel of images before feeding them to `yuv_to_rgb`.

  ```python
  yuv_images = tf.random.uniform(shape=[100, 64, 64, 3], maxval=255)
  last_dimension_axis = len(yuv_images.shape) - 1
  yuv_tensor_images = tf.truediv(
      tf.subtract(
          yuv_images,
          tf.reduce_min(yuv_images)
      ),
      tf.subtract(
          tf.reduce_max(yuv_images),
          tf.reduce_min(yuv_images)
       )
  )
  y, u, v = tf.split(yuv_tensor_images, 3, axis=last_dimension_axis)
  target_uv_min, target_uv_max = -0.5, 0.5
  u = u * (target_uv_max - target_uv_min) + target_uv_min
  v = v * (target_uv_max - target_uv_min) + target_uv_min
  preprocessed_yuv_images = tf.concat([y, u, v], axis=last_dimension_axis)
  rgb_tensor_images = tf.image.yuv_to_rgb(preprocessed_yuv_images)
  ```

  Args:
    images: 2-D or higher rank. Image data to convert. Last dimension must be
      size 3.

  Returns:
    images: tensor with the same shape as `images`.
  """
    images = ops.convert_to_tensor(images, name='images')
    kernel = ops.convert_to_tensor(_yuv_to_rgb_kernel, dtype=images.dtype, name='kernel')
    ndims = images.get_shape().ndims
    return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])