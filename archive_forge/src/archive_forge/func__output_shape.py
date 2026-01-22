from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import re
import tensorflow.compat.v2 as tf
from keras.src.engine.base_layer import Layer
from keras.src.saving import serialization_lib
def _output_shape(self, input_shape, num_elements):
    """Computes expected output shape of the dense tensor of the layer.

        Args:
          input_shape: Tensor or array with batch shape.
          num_elements: Size of the last dimension of the output.

        Returns:
          Tuple with output shape.
        """
    raise NotImplementedError('Calling an abstract method.')