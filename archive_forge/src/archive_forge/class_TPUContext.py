from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from contextlib import contextmanager
import copy
import tensorflow as tf
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.tpu import device_assignment as tpu_device_assignment
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import tpu_config
class TPUContext(object):
    """A context that holds the current configuration of the TPU computation.

  TPUContext was designed for getting TPU context information when calling
  input_fn. It can be called in model_fn as well.

  User is not expected to construct the instance from constructor. The only
  legitimate way to get the instance is either in `input_fn`:

  ```
  def input_fn(params):
    batch_size = params['batch_size']
    context = params['context']
    # ...
  ```

  or in `model_fn`

  ```
  def model_fn(params):
    batch_size = params['batch_size']
    context = params['context']
    # ...
  ```

  Most of the fields of TPUContext are useful for both `input_fn` and
  `model_fn`. Exceptions are:

  1. `input_fn` only:

    current_input_fn_deployment
    current_host

  2. `model_fn` only:

    device_assignment

  """

    def __init__(self, internal_ctx, input_device=None, invocation_index=None, call_from_input_fn=True, host_id=None):
        self._internal_ctx = internal_ctx
        self._input_device = input_device
        self._invocation_index = invocation_index
        self._call_from_input_fn = call_from_input_fn
        self._host_id = host_id

    def current_input_fn_deployment(self):
        """The configuration of the current input_fn invocation.

    The configuration depends on `TPUConfig.per_host_input_for_training`. See
    `TPUConfig` for details.

    Only set in params dict of input_fn

    Returns:
      A tuple of
        1. Device spec string: String, is the current CPU host where the
           input_fn is invoked.
        2. Current invocation index: Int, 0-based index of the input_fn
           invocation. See next item for details.
        3. Total invocation count: Int, the total number of times to invoke the
           input_fn on all CPU hosts. Each invocation will be passed with a new
           `TPUContext` instance with current invocation index set properly.
        4. Total number of replicas consumed by current_invocation: Int, the
           number of replicas fed by the data returned by current input_fn. For
           example, for per_core input pipeline deployment
           and non-model-parallelism, total invocation count is equal to
           the number of cores in the system and num replicas consumed by
           current invocation is 1. For per-host v2 input pipeline deployment,
           total invocation count is equal to the number of hosts in the system
           and num replicas consumed by current invocation is equal to number of
           replicas per host.

    Raises:
      RuntimeError: If this method is not be called from input_fn.
    """
        if not self._call_from_input_fn:
            raise RuntimeError('This TPUContext instance must not be called from model_fn.')
        if self._internal_ctx.is_input_sharded_per_core():
            total_invocation_count = self._internal_ctx.num_hosts * self._internal_ctx.num_of_replicas_per_host
            replicas_consumed = 1
        elif self._internal_ctx.is_input_broadcast_with_iterators():
            total_invocation_count = 1
            replicas_consumed = self._internal_ctx.num_replicas
        elif self._internal_ctx.is_replica_across_hosts():
            total_invocation_count = self._internal_ctx.num_replicas
            replicas_consumed = 1
        else:
            total_invocation_count = self._internal_ctx.num_hosts
            replicas_consumed = self._internal_ctx.num_of_replicas_per_host
        return (self._input_device, self._invocation_index, total_invocation_count, replicas_consumed)

    @property
    def num_replicas(self):
        """The total number of replicas.

    For non-model-parallelism, num_replicas should be the total num of TPU
    cores in the system.

    Returns:
      The number of replicas.
    """
        return self._internal_ctx.num_replicas

    @property
    def num_hosts(self):
        """The number of hosts for the TPU system."""
        return self._internal_ctx.num_hosts

    @property
    def current_host(self):
        """The current host index for the TPU system.

    Returns:
      The host index (int).

    Raises:
      RuntimeError: If this method is not be called from input_fn.
    """
        if not self._call_from_input_fn:
            raise RuntimeError('This TPUContext instance must not be called from model_fn.')
        return self._host_id

    @property
    def num_of_replicas_per_host(self):
        """The number of replicas for each host."""
        if self._internal_ctx.model_parallelism_enabled:
            raise ValueError('num_of_replicas_per_host is not supported for model_parallelism')
        return self._internal_ctx.num_of_replicas_per_host

    @property
    def device_assignment(self):
        """Returns device_assignment object.

    Raises:
      RuntimeError: If this method is not be called from model_fn.
    """
        if self._call_from_input_fn:
            raise RuntimeError('This TPUContext instance must not be called from input_fn.')
        return self._internal_ctx.device_assignment

    def device_for_replica(self, replica_id):
        """Returns the tuple of (CPU device and device ordinal) for replica.

    This should be used for full replicate for non-model-parallelism.

    Args:
       replica_id: Int, the replica index.

    Returns:
       A tuple of device spec for CPU device and int device ordinal.
    """
        return self._internal_ctx.device_for_replica(replica_id)

    @property
    def tpu_host_placement_function(self):
        """Returns the TPU host place function.

    The place function takes host_id as the input and returns the TF device
    for the correspoding host.
    """

        def _placement_function(host_id):
            """Return the host device given host_id."""
            return self._internal_ctx.tpu_host_placement_function(host_id=host_id)
        return _placement_function