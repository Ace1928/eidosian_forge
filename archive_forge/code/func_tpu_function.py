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
def tpu_function(args, kwargs):
    """TF Function used to replicate the user computation."""
    logging.vlog(1, '`TPUStrategy.run` is called with [args: %s] [kwargs: %s]', args, kwargs)
    if kwargs is None:
        kwargs = {}
    result = [[]]

    def replicated_fn(replica_id, replica_args, replica_kwargs):
        """Wraps user function to provide replica ID and `Tensor` inputs."""
        with _TPUReplicaContext(strategy, replica_id_in_sync_group=replica_id):
            result[0] = fn(*replica_args, **replica_kwargs)
        return result[0]
    replicate_inputs = []
    for i in range(strategy.num_replicas_in_sync):
        replicate_inputs.append([constant_op.constant(i, dtype=dtypes.int32), distribute_utils.select_replica(i, args), distribute_utils.select_replica(i, kwargs)])
    if options.experimental_enable_dynamic_batch_size and replicate_inputs:
        maximum_shapes = []
        flattened_list = nest.flatten(replicate_inputs[0])
        for input_tensor in flattened_list:
            if tensor_util.is_tf_type(input_tensor):
                rank = input_tensor.shape.rank
            else:
                rank = np.ndim(input_tensor)
            if rank is None:
                raise ValueError('input tensor {} to TPUStrategy.run() has unknown rank, which is not allowed'.format(input_tensor))
            maximum_shape = tensor_shape.TensorShape([None] * rank)
            maximum_shapes.append(maximum_shape)
        maximum_shapes = nest.pack_sequence_as(replicate_inputs[0], maximum_shapes)
    else:
        maximum_shapes = None
    if options.experimental_bucketizing_dynamic_shape:
        padding_spec = tpu.PaddingSpec.POWER_OF_TWO
    else:
        padding_spec = None
    with strategy.scope():
        xla_options = options.experimental_xla_options or tpu.XLAOptions(use_spmd_for_xla_partitioning=self._use_spmd_for_xla_partitioning)
        replicate_outputs = tpu.replicate(replicated_fn, replicate_inputs, device_assignment=self._device_assignment, maximum_shapes=maximum_shapes, padding_spec=padding_spec, xla_options=xla_options)
    filter_ops = lambda x: [o for o in x if not isinstance(o, ops.Operation)]
    if isinstance(result[0], list):
        result[0] = filter_ops(result[0])
    if result[0] is None or isinstance(result[0], ops.Operation):
        replicate_outputs = [None] * len(replicate_outputs)
    else:
        replicate_outputs = [nest.pack_sequence_as(result[0], filter_ops(nest.flatten(output))) for output in replicate_outputs]
    return distribute_utils.regroup(replicate_outputs)