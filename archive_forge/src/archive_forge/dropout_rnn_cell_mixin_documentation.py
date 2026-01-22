import tensorflow.compat.v2 as tf
from tensorflow.tools.docs import doc_controls
from keras.src import backend
Get the recurrent dropout mask for RNN cell.

        It will create mask based on context if there isn't any existing cached
        mask. If a new mask is generated, it will update the cache in the cell.

        Args:
          inputs: The input tensor whose shape will be used to generate dropout
            mask.
          training: Boolean tensor, whether its in training mode, dropout will
            be ignored in non-training mode.
          count: Int, how many dropout mask will be generated. It is useful for
            cell that has internal weights fused together.
        Returns:
          List of mask tensor, generated or cached mask based on context.
        