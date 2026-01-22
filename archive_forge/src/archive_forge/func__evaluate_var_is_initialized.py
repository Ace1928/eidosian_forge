import dataclasses
import functools
import os
import threading
import types as types_lib
import weakref
from google.protobuf import text_format as _text_format
from google.protobuf.message import DecodeError
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python.distribute.parallel_device import parallel_device
from tensorflow.python.eager import context
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager import monitoring
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import autograph_util
from tensorflow.python.eager.polymorphic_function import compiler_ir
from tensorflow.python.eager.polymorphic_function import eager_function_run
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import tf_method_target
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.tf_export import tf_export
def _evaluate_var_is_initialized(variables):
    """Compute booleans indicating whether each variable is initialized."""
    with ops.init_scope():
        var_is_initialized = []
        for v in variables:
            var_is_initialized.append(resource_variable_ops.var_is_initialized_op(v.handle))
        try:
            return array_ops_stack.stack(var_is_initialized).numpy()
        except errors.UnimplementedError:
            for index, v in enumerate(variables):
                try:
                    numpy_value = var_is_initialized[index].numpy()
                except errors.UnimplementedError:
                    components = parallel_device.unpack(var_is_initialized[index])
                    with ops.device(None):
                        components = array_ops_stack.stack(components)
                        all_initialized = math_ops.reduce_all(components).numpy()
                        any_initialized = math_ops.reduce_any(components).numpy()
                    if all_initialized != any_initialized:
                        raise NotImplementedError(f"Some but not all components of a parallel variable {v!r} were initialized between their creation in a tf.function and the function's trace having completed. This is not supported; consider initializing either all or none of the components, or moving initialization out of the function.")
                    numpy_value = all_initialized
                var_is_initialized[index] = numpy_value
    return var_is_initialized