import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.layers.rnn.base_wrapper import Wrapper
from keras.src.utils import generic_utils
from keras.src.utils import layer_utils
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
Computes an output mask tensor for Embedding layer.

        This is based on the inputs, mask, and the inner layer.
        If batch size is specified:
        Simply return the input `mask`. (An rnn-based implementation with
        more than one rnn inputs is required but not supported in tf.keras yet.)
        Otherwise we call `compute_mask` of the inner layer at each time step.
        If the output mask at each time step is not `None`:
        (E.g., inner layer is Masking or RNN)
        Concatenate all of them and return the concatenation.
        If the output mask at each time step is `None` and the input mask is not
        `None`:(E.g., inner layer is Dense)
        Reduce the input_mask to 2 dimensions and return it.
        Otherwise (both the output mask and the input mask are `None`):
        (E.g., `mask` is not used at all)
        Return `None`.

        Args:
          inputs: Tensor with shape [batch size, timesteps, ...] indicating the
            input to TimeDistributed. If static shape information is available
            for "batch size", `mask` is returned unmodified.
          mask: Either None (indicating no masking) or a Tensor indicating the
            input mask for TimeDistributed. The shape can be static or dynamic.

        Returns:
          Either None (no masking), or a [batch size, timesteps, ...] Tensor
          with an output mask for the TimeDistributed layer with the shape
          beyond the second dimension being the value of the input mask shape(if
          the computed output mask is none), an output mask with the shape
          beyond the first dimension being the value of the mask shape(if mask
          is not None) or output mask with the shape beyond the first dimension
          being the value of the computed output shape.

        