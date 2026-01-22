import abc
from typing import Any, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.ragged.row_partition import RowPartitionSpec
from tensorflow.python.types import core
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _get_layer_broadcasters_from_rps(zero_broadcaster, source_rps, target_rps):
    """Get LayerBroadcasters from RowPartitions.

           *--zero_broadcaster->*
           |                    |
         source_rps[0]     target_rps[0]
           |                    |
           V                    V
           *---result[1]------->*
           |                    |
         source_rps[1]     target_rps[1]
           |                    |
           V                    V
           *---result[2]------->*
                  .
                  .
                  .
           *---result[k-1]----->*
           |                    |
         source_rps[k]     target_rps[k]
           |                    |
           V                    V
           *---result[k]------->*

  Note: result[0] = zero_broadcaster

  Args:
    zero_broadcaster: a broadcaster between the source and target row
      partitions' rows, and equal to result[0].
    source_rps: source row partitions.
    target_rps: target row partitions (same length as source_rps).

  Returns:
    result: a list of LayerBroadcasters.
  """
    if not isinstance(zero_broadcaster, _LayerBroadcaster):
        raise TypeError('Not a _LayerBroadcaster: ' + str(zero_broadcaster))
    assert len(source_rps) == len(target_rps)
    if not source_rps:
        return [zero_broadcaster]
    next_broadcaster = zero_broadcaster.next_layer(source_rps[0], target_rps[0])
    tail_broadcasters = _get_layer_broadcasters_from_rps(next_broadcaster, source_rps[1:], target_rps[1:])
    return [zero_broadcaster] + tail_broadcasters