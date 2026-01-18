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
def unwrap_values(distribution_strategy, grouped_inputs, grouped_outputs, grouped_updates=None, grouped_session_args=None, with_loss_tensor=False):
    """Unwrap the list of values contained in the PerReplica parameters.

  This function calls `flatten_per_replica_values` to parse each of the input
  parameters into a list of values on the different devices. If we set
  `with_loss_tensor` to be True, we also call `reduce` on the list of losses on
  the different devices to give us one loss tensor.

  Args:
    distribution_strategy: DistributionStrategy used to distribute training and
        validation.
    grouped_inputs: PerReplica inputs returned from the train or test function
        that we ran on each device.
    grouped_outputs: PerReplica outputs returned from the train or test function
        that we ran on each device.
    grouped_updates: PerReplica updates returned from the train or test function
        that we ran on each device.
    grouped_session_args: PerReplica session args returned from the train or
        test function that we ran on each device.
    with_loss_tensor: Boolean that indicates if we need to add the reduced loss
        tensor as one of the outputs.

  Returns:
    Values of each of the PerReplica parameters.

  """
    all_inputs = flatten_per_replica_values(distribution_strategy, grouped_inputs)
    all_outputs = unwrap_outputs(distribution_strategy, grouped_outputs, with_loss_tensor)
    if grouped_updates:
        all_updates = flatten_per_replica_values(distribution_strategy, grouped_updates)
    else:
        all_updates = None
    all_session_args = {}
    if grouped_session_args:
        grouped_feed_dict = grouped_session_args.get('feed_dict')
        if grouped_feed_dict:
            all_session_args['feed_dict'] = flatten_per_replica_values(distribution_strategy, grouped_feed_dict)
        grouped_fetches = grouped_session_args.get('fetches')
        if grouped_fetches:
            all_session_args['fetches'] = flatten_per_replica_values(distribution_strategy, grouped_fetches)
    return (all_inputs, all_outputs, all_updates, all_session_args)