import collections
import contextlib
import copy
import enum  # pylint: disable=g-bad-import-order
import functools
import threading
import weakref
import six
from tensorflow.python import tf2
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context as eager_context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@tf_export('__internal__.distribute.variable_sync_on_read_context', v1=[])
@contextlib.contextmanager
def variable_sync_on_read_context():
    """A context that forces SyncOnReadVariable to aggregate upon reading.

  This context is useful if one wants to read the aggregated value out of a
  SyncOnReadVariable in replica context. By default the aggregation is turned
  off per the definition of SyncOnReadVariable.

  When reading a SyncOnReadVariable in cross-replica context, aggregation is
  always turned on so there is no need for such context.

  By reading a SyncOnReadVariable, we mean:
    1. Convert the variable to a tensor using `convert_to_tensor`.
    2. Calling `variable.value()` or `variable.read_value()`.

  Example usage:

  ```
  strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
  with strategy.scope():
    v = tf.Variable(1.0, synchronization=tf.VariableSynchronization.ON_READ,
      aggregation=tf.VariableAggregation.SUM)

  def replica_fn():
    return v + 10.0

  non_aggregated = strategy.run(replica_fn)
  print(non_aggregated) # PerReplica: {0: 11.0, 1: 11.0}

  def replica_fn():
    with variable_sync_on_read_context():
      return v + 10.0

  aggregated = strategy.run(replica_fn)
  print(aggregated) # PerReplica: {0: 12.0, 1: 12.0}
  ```

  Yields:
    Context manager for aggregating SyncOnReadVariable upon reading.
  """
    try:
        _variable_sync_on_read_context.entered = True
        yield
    finally:
        _variable_sync_on_read_context.entered = False