import functools
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.distribute import distributed_training_utils as dist_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def unwrap_output_dict(strategy, grouped_outputs, mode):
    """Unwrap the list of outputs contained in the PerReplica parameters."""
    if mode == ModeKeys.PREDICT:
        return flatten_per_replica_values(strategy, grouped_outputs)
    total_loss = strategy.reduce(reduce_util.ReduceOp.SUM, grouped_outputs['total_loss'][0], axis=None)
    output_losses = flatten_per_replica_values(strategy, grouped_outputs['output_losses'])
    metrics = flatten_per_replica_values(strategy, grouped_outputs['metrics'])
    batch_size = strategy.reduce(reduce_util.ReduceOp.SUM, grouped_outputs['batch_size'], axis=None)
    if backend.is_tpu_strategy(strategy) and ops.executing_eagerly_outside_functions():
        output_losses = output_losses[::strategy.num_replicas_in_sync]
        metrics = metrics[::strategy.num_replicas_in_sync]
    return {'total_loss': [total_loss], 'output_losses': output_losses, 'metrics': metrics, 'batch_size': batch_size}