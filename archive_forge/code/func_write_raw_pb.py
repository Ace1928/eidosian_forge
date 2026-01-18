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
@tf_export('summary.experimental.write_raw_pb', v1=[])
def write_raw_pb(tensor, step=None, name=None):
    """Writes a summary using raw `tf.compat.v1.Summary` protocol buffers.

  Experimental: this exists to support the usage of V1-style manual summary
  writing (via the construction of a `tf.compat.v1.Summary` protocol buffer)
  with the V2 summary writing API.

  Args:
    tensor: the string Tensor holding one or more serialized `Summary` protobufs
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.
    name: Optional string name for this op.

  Returns:
    True on success, or false if no summary was written because no default
    summary writer was available.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
    with ops.name_scope(name, 'write_raw_pb') as scope:
        if _summary_state.writer is None:
            return constant_op.constant(False)
        if step is None:
            step = get_step()
            if step is None:
                raise ValueError('No step set. Please specify one either through the `step` argument or through tf.summary.experimental.set_step()')

        def record():
            """Record the actual summary and return True."""
            with ops.device('cpu:0'):
                raw_summary_op = gen_summary_ops.write_raw_proto_summary(_summary_state.writer._resource, step, array_ops.identity(tensor), name=scope)
                with ops.control_dependencies([raw_summary_op]):
                    return constant_op.constant(True)
        with ops.device('cpu:0'):
            op = smart_cond.smart_cond(should_record_summaries(), record, _nothing, name='summary_cond')
            if not context.executing_eagerly():
                ops.add_to_collection(ops.GraphKeys._SUMMARY_COLLECTION, op)
            return op