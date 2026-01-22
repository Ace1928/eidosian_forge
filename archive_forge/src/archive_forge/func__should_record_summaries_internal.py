import abc
import collections
import functools
import os
import re
import threading
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import profiler as _profiler
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_summary_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import summary_op_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import resource
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
def _should_record_summaries_internal(default_state):
    """Returns boolean Tensor if summaries should/shouldn't be recorded.

  Now the summary condition is decided by logical "and" of below conditions:
  First, summary writer must be set. Given this constraint is met,
  ctx.summary_recording and ctx.summary_recording_distribution_strategy.
  The former one is usually set by user, and the latter one is controlled
  by DistributionStrategy (tf.distribute.ReplicaContext).

  Args:
    default_state: can be True or False. The default summary behavior when
    summary writer is set and the user does not specify
    ctx.summary_recording and ctx.summary_recording_distribution_strategy
    is True.
  """
    if _summary_state.writer is None:
        return constant_op.constant(False)
    if not callable(_summary_state.is_recording):
        static_cond = tensor_util.constant_value(_summary_state.is_recording)
        if static_cond is not None and (not static_cond):
            return constant_op.constant(False)
    resolve = lambda x: x() if callable(x) else x
    cond_distributed = resolve(_summary_state.is_recording_distribution_strategy)
    cond = resolve(_summary_state.is_recording)
    if cond is None:
        cond = default_state
    return math_ops.logical_and(cond_distributed, cond)