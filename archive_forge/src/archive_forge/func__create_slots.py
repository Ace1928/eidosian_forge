from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
from six.moves import range
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework.ops import internal_convert_to_tensor
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.ops import gen_sdca_ops
from tensorflow.python.ops import variables as var_ops
from tensorflow.python.ops.nn import log_poisson_loss
from tensorflow.python.ops.nn import sigmoid_cross_entropy_with_logits
from tensorflow_estimator.python.estimator.canned.linear_optimizer.python.utils.sharded_mutable_dense_hashtable import _ShardedMutableDenseHashTable
def _create_slots(self):
    """Make unshrunk internal variables (slots)."""
    self._slots = collections.defaultdict(list)
    for name in ['sparse_features_weights', 'dense_features_weights']:
        for var in self._variables[name]:
            if isinstance(var, var_ops.PartitionedVariable) or isinstance(var, list):
                var_list = []
                for v in var:
                    with ops.colocate_with(v):
                        slot_var = tf.Variable(initial_value=tf.compat.v1.zeros_like(tf.cond(tf.compat.v1.is_variable_initialized(v), v.read_value, lambda: v.initial_value), tf.dtypes.float32), name=v.op.name + '_unshrunk')
                        var_list.append(slot_var)
                self._slots['unshrunk_' + name].append(var_list)
            else:
                with tf.compat.v1.device(var.device):
                    self._slots['unshrunk_' + name].append(tf.Variable(tf.compat.v1.zeros_like(tf.cond(tf.compat.v1.is_variable_initialized(var), var.read_value, lambda: var.initial_value), tf.dtypes.float32), name=var.op.name + '_unshrunk'))