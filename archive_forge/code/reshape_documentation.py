import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.engine.base_layer import Layer
from tensorflow.python.util.tf_export import keras_export
Find and replace a missing dimension in an output shape.

        This is a near direct port of the internal Numpy function
        `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`

        Args:
          input_shape: Shape of array being reshaped
          output_shape: Desired shape of the array with at most a single -1
            which indicates a dimension that should be derived from the input
            shape.

        Returns:
          The new output shape with a -1 replaced with its computed value.

        Raises:
          ValueError: If the total array size of the output_shape is
          different than the input_shape, or more than one unknown dimension
          is specified.
        