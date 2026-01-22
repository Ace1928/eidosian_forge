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
def _next_instance_key(self):
    """Returns the next instance key."""
    if self._use_unique_instance_key():
        graph = ops.get_default_graph()
        while getattr(graph, 'is_control_flow_graph', False):
            graph = graph.outer_graph
        if not context.executing_eagerly() and graph.building_function:
            with graph.as_default():
                return graph.capture_call_time_value(self._next_instance_key, tensor_spec.TensorSpec([], dtypes.int32))
        else:
            instance_key = self._collective_keys.get_instance_key(self._group_key, self._device)
            with ops.device('CPU:0'):
                return ops.convert_to_tensor(instance_key, dtype=dtypes.int32)
    else:
        return self._collective_keys.get_instance_key(self._group_key, self._device)