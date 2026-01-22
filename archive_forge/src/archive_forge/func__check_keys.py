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
def _check_keys(self, keys):
    if keys.get_shape().ndims != 1 and keys.get_shape().ndims != 2:
        raise ValueError('Expected a vector or matrix for keys, got %s.' % keys.get_shape())