import atexit
import collections
import contextlib
import copy
import functools
import weakref
from absl import logging
import numpy as np
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_replicated_variable
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver as tpu_cluster_resolver_lib
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import save_context
from tensorflow.python.tpu import device_assignment as device_assignment_lib  # pylint: disable=unused-import
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_hardware_feature
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export('distribute.experimental.TPUStrategy', v1=[])
@deprecation.deprecated_endpoints('distribute.experimental.TPUStrategy')
class TPUStrategy(distribute_lib.Strategy):
    """Synchronous training on TPUs and TPU Pods.

  To construct a TPUStrategy object, you need to run the
  initialization code as below:

  >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  >>> tf.config.experimental_connect_to_cluster(resolver)
  >>> tf.tpu.experimental.initialize_tpu_system(resolver)
  >>> strategy = tf.distribute.experimental.TPUStrategy(resolver)

  While using distribution strategies, the variables created within the
  strategy's scope will be replicated across all the replicas and can be kept in
  sync using all-reduce algorithms.

  To run TF2 programs on TPUs, you can either use `.compile` and
  `.fit` APIs in `tf.keras` with TPUStrategy, or write your own customized
  training loop by calling `strategy.run` directly. Note that
  TPUStrategy doesn't support pure eager execution, so please make sure the
  function passed into `strategy.run` is a `tf.function` or
  `strategy.run` is called inside a `tf.function` if eager
  behavior is enabled.
  """

    def __init__(self, tpu_cluster_resolver=None, device_assignment=None):
        """Synchronous training in TPU donuts or Pods.

    Args:
      tpu_cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
        which provides information about the TPU cluster.
      device_assignment: Optional `tf.tpu.experimental.DeviceAssignment` to
        specify the placement of replicas on the TPU cluster.
    """
        logging.warning('`tf.distribute.experimental.TPUStrategy` is deprecated, please use the non-experimental symbol `tf.distribute.TPUStrategy` instead.')
        super(TPUStrategy, self).__init__(TPUExtended(self, tpu_cluster_resolver, device_assignment=device_assignment, enable_data_reorder=device_assignment is not None))
        distribute_lib.distribution_strategy_gauge.get_cell('V2').set('TPUStrategy')
        distribute_lib.distribution_strategy_replica_gauge.get_cell('num_workers').set(self.extended.num_hosts)
        distribute_lib.distribution_strategy_replica_gauge.get_cell('num_replicas_per_worker').set(self.extended.num_replicas_per_host)
        self._enable_packed_variable_in_eager_mode = True

    def run(self, fn, args=(), kwargs=None, options=None):
        """See base class."""
        validate_run_function(fn)
        fn, args, kwargs = _maybe_partial_apply_variables(fn, args, kwargs)
        fn = autograph.tf_convert(fn, autograph_ctx.control_status_ctx())
        options = options or distribute_lib.RunOptions()
        return self.extended.tpu_run(fn, args, kwargs, options)

    @property
    def cluster_resolver(self):
        """Returns the cluster resolver associated with this strategy.

    `tf.distribute.experimental.TPUStrategy` provides the
    associated `tf.distribute.cluster_resolver.ClusterResolver`. If the user
    provides one in `__init__`, that instance is returned; if the user does
    not, a default
    `tf.distribute.cluster_resolver.TPUClusterResolver` is provided.
    """
        return self.extended._tpu_cluster_resolver