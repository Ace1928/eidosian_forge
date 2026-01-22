import collections
import hashlib
import operator
import os
import os.path
import sys
import numpy as np
from tensorflow.core.framework import summary_pb2
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import analytics
from tensorflow.python.platform import gfile
from tensorflow.python.platform import remote_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary_iterator
from tensorflow.python.tpu import tensor_tracer_flags
from tensorflow.python.tpu import tensor_tracer_report
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import training_util
def _write_cache(step, event_file_suffix=None, **kwargs):
    """Writes the given caches as tensor summary.

      Args:
        step: Step tensor with dimension [num_cores].
        event_file_suffix: Event filename suffix tensor.
        **kwargs: The dictionary of tensors that needs to be written as
          summaries. Key and value pairs within kwargs correspond to the tag
          name, and tensor content that will be written using summary.write.
          The trace_modes that use this function are:
            - summary: In summary mode, kwargs includes a single (tag, content)
            pair which are, _TT_SUMMARY_TAG and a tf.float32 signature_cache
            variable. The dimension of the signature_cache is:
              num_cores x num_traced_tensors x num_signatures.
            - full_tensor_summary: kwargs will include all traced tensors. Tag
            and content correspond to the name of the tensor, and its actual
            content.
      Returns:
        A tf.Operation that needs to be executed for the host call dependencies.
      """
    file_suffix = _TT_EVENT_FILE_SUFFIX
    if event_file_suffix is not None:
        file_suffix = string_ops.string_join([file_suffix, event_file_suffix], separator='.')
    summary_write_ops = []
    summary_writer = summary.create_file_writer_v2(self._parameters.trace_dir, filename_suffix=file_suffix, max_queue=_TT_SUMMARY_MAX_QUEUE)
    graph.add_to_collection(TENSOR_TRACER_SUMMARY_COLLECTION, summary_writer)
    step_value = step[0]
    dt = step_value.dtype
    if dt.__ne__(dtypes.int64) and dt.__ne__(dtypes.uint64) and dt.__ne__(dtypes.float64):
        step_value = math_ops.cast(step_value, dtypes.int64)
    with summary_writer.as_default():
        summary_metadata = summary_pb2.SummaryMetadata(plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name=_TT_TENSORBOARD_PLUGIN_NAME))
        for key, value in kwargs.items():
            if not self._parameters.collect_summary_per_core:
                if key == _TT_SUMMARY_TAG and value.shape.as_list()[0] != 1:
                    value = self.aggregate_global_cache(value)
            with ops.control_dependencies([summary_writer.init()]):
                summary_write_ops.append(summary.write(_TT_SUMMARY_TAG + '/' + key + '.' + graph_summary_tag, value, metadata=summary_metadata, step=step_value))
    return control_flow_ops.group(summary_write_ops)