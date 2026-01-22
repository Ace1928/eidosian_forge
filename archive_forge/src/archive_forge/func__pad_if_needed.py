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
def _pad_if_needed(batch_key_item, fillvalue=0):
    """ Returns padded batch.

  Args:
    batch_key_item: List of data samples of any type with shape
      [batch_size, ..., padded_dim(None)].
    fillvalue: Default fillvalue to use.

  Returns:
    Padded with zeros tensor of same type and shape
      [batch_size, ..., max_padded_dim_len].

  Raises:
    ValueError if data samples have different shapes (except last padded dim).
  """
    shapes = [seq.shape[:-1] if len(seq.shape) > 0 else -1 for seq in batch_key_item]
    if not all((shapes[0] == x for x in shapes)):
        raise ValueError('Array shapes must match.')
    last_length = [seq.shape[-1] if len(seq.shape) > 0 else 0 for seq in batch_key_item]
    if all([x == last_length[0] for x in last_length]):
        return batch_key_item
    batch_size = len(batch_key_item)
    max_sequence_length = max(last_length)
    result_batch = np.zeros(shape=[batch_size] + list(shapes[0]) + [max_sequence_length], dtype=batch_key_item[0].dtype)
    _fill_array(result_batch, batch_key_item, fillvalue)
    return result_batch