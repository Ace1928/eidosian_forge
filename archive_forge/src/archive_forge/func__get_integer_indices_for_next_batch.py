from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import random
import types as tp
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator.inputs.queues import feeding_queue_runner as fqr
def _get_integer_indices_for_next_batch(batch_indices_start, batch_size, epoch_end, array_length, current_epoch, total_epochs):
    """Returns the integer indices for next batch.

  If total epochs is not None and current epoch is the final epoch, the end
  index of the next batch should not exceed the `epoch_end` (i.e., the final
  batch might not have size `batch_size` to avoid overshooting the last epoch).

  Args:
    batch_indices_start: Integer, the index to start next batch.
    batch_size: Integer, size of batches to return.
    epoch_end: Integer, the end index of the epoch. The epoch could start from a
      random position, so `epoch_end` provides the end index for that.
    array_length: Integer, the length of the array.
    current_epoch: Integer, the epoch number has been emitted.
    total_epochs: Integer or `None`, the total number of epochs to emit. If
      `None` will run forever.

  Returns:
    A tuple of a list with integer indices for next batch and `current_epoch`
    value after the next batch.

  Raises:
    OutOfRangeError if `current_epoch` is not less than `total_epochs`.

  """
    if total_epochs is not None and current_epoch >= total_epochs:
        raise tf.errors.OutOfRangeError(None, None, 'Already emitted %s epochs.' % current_epoch)
    batch_indices_end = batch_indices_start + batch_size
    batch_indices = [j % array_length for j in range(batch_indices_start, batch_indices_end)]
    epoch_end_indices = [i for i, x in enumerate(batch_indices) if x == epoch_end]
    current_epoch += len(epoch_end_indices)
    if total_epochs is None or current_epoch < total_epochs:
        return (batch_indices, current_epoch)
    final_epoch_end_inclusive = epoch_end_indices[-(current_epoch - total_epochs + 1)]
    batch_indices = batch_indices[:final_epoch_end_inclusive + 1]
    return (batch_indices, total_epochs)