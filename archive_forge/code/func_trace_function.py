import contextlib
import dataclasses
import enum
import threading
from typing import Any, Callable, Dict, Optional, Tuple
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.core.function.polymorphism import function_cache as function_cache_lib
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.eager import monitoring
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import concrete_function as concrete_function_lib
from tensorflow.python.eager.polymorphic_function import function_context
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import transform
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.util import compat
def trace_function(args=None, kwargs=None, tracing_options=None):
    """Returns a `ConcreteFunction` specialized to inputs and execution context.

  Compiles a Graph corresponding to the Python function logic and uses that
  to generate a differentiable ConcreteFunction.

  Args:
    args: inputs to specialize on. Can be concrete values (e.g. 1) or
      `tf.Tensor` or `tf.TensorSpec`.
    kwargs: keyword inputs to specialize on. Concrete values (e.g. 1) or
      `tf.Tensor` or `tf.TensorSpec`.
    tracing_options: TracingOptions for the tracing process.
  """
    if not tracing_options:
        tracing_options = TracingOptions()
    args = args if args else ()
    kwargs = kwargs if kwargs else {}
    if tracing_options.input_signature and (args or kwargs):
        bound_args = function_type_utils.bind_function_inputs(args, kwargs, tracing_options.polymorphic_type, tracing_options.default_values)
        args, kwargs = (bound_args.args, bound_args.kwargs)
    with tracing_options.lock or contextlib.nullcontext():
        if tracing_options.input_signature and (not args) and (not kwargs):
            args = tracing_options.input_signature
            kwargs = {}
        concrete_function = _maybe_define_function(args, kwargs, tracing_options)
        _set_arg_keywords(concrete_function)
    if not tracing_options.bind_graph_to_function:
        concrete_function._garbage_collector.release()
    return concrete_function