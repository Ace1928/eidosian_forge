import copy
import threading
from typing import Callable, List, Optional, Union
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
def aggregate_gradients_using_hierarchical_copy(avail_devices, replica_grads):
    """Aggregate gradients using hierarchical copies.

  Args:
    avail_devices: available GPU devices.
    replica_grads: List of lists of (gradient, variable) tuples. The outer list
      is over replicas. The inner list is over individual gradients.

  Returns:
    The list of (aggregated_gradient, variable), where the gradient has been
      summed across all replicas and the variable is chosen from the first
      replica.
  """
    agg_grads = []
    num_devices = len(avail_devices)
    group_size = num_devices // 2
    for i, single_grads in enumerate(zip(*replica_grads)):
        group_0_main_device = i % num_devices
        group_1_main_device = (group_0_main_device + group_size) % num_devices
        if group_0_main_device < group_size:
            group_0_begin = 0
            group_1_begin = group_size
        else:
            group_0_begin = group_size
            group_1_begin = 0
        group_0_device_grads = single_grads[group_0_begin:group_0_begin + group_size]
        with ops.device(avail_devices[group_0_main_device]):
            group_0_agg_grads, _ = aggregate_single_gradient_using_copy(group_0_device_grads, False, False)
        group_1_device_grads = single_grads[group_1_begin:group_1_begin + group_size]
        with ops.device(avail_devices[group_1_main_device]):
            group_1_agg_grads, _ = aggregate_single_gradient_using_copy(group_1_device_grads, False, False)
        with ops.device(avail_devices[group_0_main_device]):
            (agg_total_grads, _), _ = aggregate_single_gradient_using_copy([group_0_agg_grads, group_1_agg_grads], False, False)
        with ops.device(avail_devices[group_0_main_device]):
            group_0_agg_grads_bcast = array_ops.identity(agg_total_grads)
        with ops.device(avail_devices[group_1_main_device]):
            group_1_agg_grads_bcast = array_ops.identity(agg_total_grads)
        agg_grads_bcast = []
        for j in range(len(single_grads)):
            with ops.device(avail_devices[j]):
                if (group_0_main_device < group_size) == (j < group_size):
                    src_device_grad = group_0_agg_grads_bcast
                else:
                    src_device_grad = group_1_agg_grads_bcast
                agg_grads_bcast.append(array_ops.identity(src_device_grad))
        agg_grads.append([(g, v) for g, (_, v) in zip(agg_grads_bcast, single_grads)])
    agg_grads = list(zip(*agg_grads))
    return agg_grads