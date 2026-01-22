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
def _create_resource(self):
    use_node_name_sharing = self._checkpoint and self._shared_name is None
    table_ref = gen_lookup_ops.mutable_dense_hash_table_v2(empty_key=self._empty_key, deleted_key=self._deleted_key, shared_name=self._shared_name, use_node_name_sharing=use_node_name_sharing, value_dtype=self._value_dtype, value_shape=self._value_shape, initial_num_buckets=self._initial_num_buckets, name=self._name)
    if tf.executing_eagerly():
        self._table_name = None
    else:
        self._table_name = table_ref.op.name.split('/')[-1]
    return table_ref