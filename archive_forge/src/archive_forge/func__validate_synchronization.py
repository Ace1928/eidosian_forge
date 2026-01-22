from collections import abc
import contextlib
import threading
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_values as tpu_values_lib
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _validate_synchronization(kwargs):
    """Validate that given synchronization value is valid."""
    synchronization = kwargs.get('synchronization', vs.VariableSynchronization.AUTO)
    if synchronization == vs.VariableSynchronization.NONE:
        raise ValueError('`NONE` variable synchronization mode is not supported with tf.distribute strategy. Please change the `synchronization` for variable: ' + str(kwargs['name']))
    if synchronization not in (vs.VariableSynchronization.ON_READ, vs.VariableSynchronization.ON_WRITE, vs.VariableSynchronization.AUTO):
        raise ValueError('Invalid variable synchronization mode: %s for variable: %s' % (synchronization, kwargs['name']))
    if synchronization == vs.VariableSynchronization.AUTO:
        return vs.VariableSynchronization.ON_WRITE
    return synchronization