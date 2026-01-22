import collections
import copy
import enum
import re
import sys
import threading
import types
from typing import Any, AnyStr, Callable, List, NoReturn, Pattern, Tuple, Type, Union, Optional
from absl import app
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import record
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import registry
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import traceable_stack
from tensorflow.python.framework import versions
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace as profiler_trace
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import lock_util
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_stack
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import kwarg_only
from tensorflow.python.util.tf_export import tf_export
def _as_graph_def(self, from_version=None, add_shapes=False, use_pybind11_proto=False):
    """Returns a serialized `GraphDef` representation of this graph.

    The serialized `GraphDef` can be imported into another `Graph`
    (using `tf.import_graph_def`) or used with the
    [C++ Session API](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.10/tensorflow/g3doc/api_docs/cc/index.md).

    This method is thread-safe.

    Args:
      from_version: Optional.  If this is set, returns a `GraphDef` containing
        only the nodes that were added to this graph since its `version`
        property had the given value.
      add_shapes: If true, adds an "_output_shapes" list attr to each node with
        the inferred shapes of each of its outputs.
      use_pybind11_proto: If true, uses the c++ pybind11_proto api to get the
        GraphDef proto directly from c++, instead of through a TF buffer.

    Returns:
      A tuple containing a
      [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
      protocol buffer, and the version of the graph to which that
      `GraphDef` corresponds.

    Raises:
      ValueError: If the `graph_def` would be too large.

    """
    with self._lock:
        if use_pybind11_proto:
            with self._c_graph.get() as c_graph:
                graph = graph_pb2.GraphDef()
                graph.CopyFrom(pywrap_tf_session.TF_GraphToGraphDefPybind(c_graph))
        else:
            with c_api_util.tf_buffer() as buf:
                with self._c_graph.get() as c_graph:
                    pywrap_tf_session.TF_GraphToGraphDef(c_graph, buf)
                    data = pywrap_tf_session.TF_GetBuffer(buf)
            graph = graph_pb2.GraphDef()
            graph.ParseFromString(compat.as_bytes(data))
        if not graph.library.function:
            graph.ClearField('library')
        if add_shapes:
            for node in graph.node:
                op = self._get_operation_by_name(node.name)
                if op.outputs:
                    node.attr['_output_shapes'].list.shape.extend([output.get_shape().as_proto() for output in op.outputs])
            for function_def in graph.library.function:
                defined_function = self._functions[function_def.signature.name]
                try:
                    func_graph = defined_function.graph
                except AttributeError:
                    continue
                input_shapes = function_def.attr['_input_shapes']
                try:
                    func_graph_inputs = func_graph.inputs
                except AttributeError:
                    continue
                assert len(input_shapes.list.shape) in [0, len(func_graph_inputs)]
                if not input_shapes.list.shape:
                    for input_tensor, arg_def in zip(func_graph_inputs, function_def.signature.input_arg):
                        input_shapes.list.shape.add().CopyFrom(input_tensor.get_shape().as_proto())
                        if input_tensor.dtype == dtypes.resource:
                            _copy_handle_data_to_arg_def(input_tensor, arg_def)
                for output_tensor, arg_def in zip(func_graph.outputs, function_def.signature.output_arg):
                    if output_tensor.dtype == dtypes.resource:
                        _copy_handle_data_to_arg_def(output_tensor, arg_def)
                for node in function_def.node_def:
                    try:
                        op = func_graph.get_operation_by_name(node.name)
                    except KeyError:
                        continue
                    outputs = op.outputs
                    if op.type == 'StatefulPartitionedCall':
                        num_outputs = len(node.attr['Tout'].list.type)
                        outputs = outputs[:num_outputs]
                    node.attr['_output_shapes'].list.shape.extend([output.get_shape().as_proto() for output in outputs])
    return (graph, self.version)