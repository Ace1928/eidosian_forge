from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from six.moves import range
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.checkpoint import saveable_compat
Returns lists of the keys and values tensors in the sharded table.

    Args:
      name: name of the table.

    Returns:
      A pair of lists with the first list containing the key tensors and the
        second list containing the value tensors from each shard.
    