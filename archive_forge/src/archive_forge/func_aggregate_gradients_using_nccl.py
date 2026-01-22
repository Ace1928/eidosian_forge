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
def aggregate_gradients_using_nccl(replica_grads):
    """Aggregate gradients using nccl allreduce."""
    agg_all_g_and_v = []
    for single_g_and_v in zip(*replica_grads):
        single_grads = [g for g, _ in single_g_and_v]
        agg_grads = nccl_ops.all_sum(single_grads)
        agg_all_g_and_v.append([(g, v) for g, (_, v) in zip(agg_grads, single_g_and_v)])
    agg_all_g_and_v = list(zip(*agg_all_g_and_v))
    return agg_all_g_and_v