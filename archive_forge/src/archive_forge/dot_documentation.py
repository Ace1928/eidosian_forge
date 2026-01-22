import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer_utils
from keras.src.layers.merging.base_merge import _Merge
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
Initializes a layer that computes the element-wise dot product.

          >>> x = np.arange(10).reshape(1, 5, 2)
          >>> print(x)
          [[[0 1]
            [2 3]
            [4 5]
            [6 7]
            [8 9]]]
          >>> y = np.arange(10, 20).reshape(1, 2, 5)
          >>> print(y)
          [[[10 11 12 13 14]
            [15 16 17 18 19]]]
          >>> tf.keras.layers.Dot(axes=(1, 2))([x, y])
          <tf.Tensor: shape=(1, 2, 2), dtype=int64, numpy=
          array([[[260, 360],
                  [320, 445]]])>

        Args:
          axes: Integer or tuple of integers,
            axis or axes along which to take the dot product. If a tuple, should
            be two integers corresponding to the desired axis from the first
            input and the desired axis from the second input, respectively. Note
            that the size of the two selected axes must match.
          normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
          **kwargs: Standard layer keyword arguments.
        