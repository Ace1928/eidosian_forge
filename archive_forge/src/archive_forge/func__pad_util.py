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
def _pad_util(input_tensor, full_axis_dim):
    """Pad the `input_tensor`'s first dimension to be `full_axis_dim`."""
    missing_axis_dim = full_axis_dim - array_ops.shape_v2(input_tensor)[0]
    tensor_rank = array_ops.rank(input_tensor)
    paddings_axis = [[0, missing_axis_dim]]
    paddings = array_ops.concat([paddings_axis, array_ops.zeros(shape=(tensor_rank - 1, 2), dtype=dtypes.int32)], axis=0)
    padded_input_tensor = array_ops.pad(input_tensor, paddings)
    return padded_input_tensor