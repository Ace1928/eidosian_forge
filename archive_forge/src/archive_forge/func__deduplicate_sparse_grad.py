import abc
import platform
import re
import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import backend
from keras.src import initializers
from keras.src.dtensor import utils as dtensor_utils
from keras.src.optimizers import utils as optimizer_utils
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
def _deduplicate_sparse_grad(self, grads):
    """Deduplicate sparse gradient.

        For sparse gradients, i.e., gradient is of type `tf.IndexedSlices`,
        it is possible that `gradient.indices` has duplicated indices.
        This function adds up values for the duplicated indices, and returns
        a `tf.IndexedSlices` with indices of unique values.
        """
    processed_grads = []
    for grad in grads:
        if isinstance(grad, tf.IndexedSlices):
            values = grad.values
            indices = grad.indices
            unique_indices, new_index_positions = tf.unique(indices)
            summed_values = tf.math.unsorted_segment_sum(values, new_index_positions, tf.shape(unique_indices)[0])
            processed_grads.append(tf.IndexedSlices(summed_values, unique_indices, grad.dense_shape))
        else:
            processed_grads.append(grad)
    return processed_grads