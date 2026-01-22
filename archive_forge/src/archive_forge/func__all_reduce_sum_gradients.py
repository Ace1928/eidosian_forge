import warnings
import tensorflow as tf
from keras.src import backend
from keras.src.backend.common import KerasVariable
from keras.src.backend.tensorflow.trackable import KerasAutoTrackable
from keras.src.optimizers import base_optimizer
def _all_reduce_sum_gradients(self, grads_and_vars):
    """Returns all-reduced gradients aggregated via summation.

        Args:
            grads_and_vars: List of (gradient, variable) pairs.

        Returns:
            List of (gradient, variable) pairs
            where gradients have been all-reduced.
        """
    replica_context = tf.distribute.get_replica_context()
    if not replica_context:
        return grads_and_vars
    grads_and_vars = list(grads_and_vars)
    filtered_grads_and_vars = filter_empty_gradients(grads_and_vars)
    if filtered_grads_and_vars:
        grads = [pair[0] for pair in filtered_grads_and_vars]
        reduced = tf.distribute.get_replica_context().all_reduce(tf.distribute.ReduceOp.SUM, grads)
    else:
        reduced = []
    reduced_with_nones = []
    reduced_pos = 0
    for g, v in grads_and_vars:
        if g is None:
            reduced_with_nones.append((None, v))
        else:
            reduced_with_nones.append((reduced[reduced_pos], v))
            reduced_pos += 1
    assert reduced_pos == len(reduced), 'Failed to add all gradients'
    return reduced_with_nones