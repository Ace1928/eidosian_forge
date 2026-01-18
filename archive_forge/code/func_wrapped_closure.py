import traceback
from typing import Any, Callable, Hashable
import weakref
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager.polymorphic_function import composite_tensor_utils
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.saved_model import save_context
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def wrapped_closure():
    if save_context.in_save_context() and default_value is not None:
        return default_value
    if not context.executing_eagerly():
        graph = ops.get_default_graph()
        assert isinstance(graph, FuncGraph), 'This API should only be used in TF2 enviroment.'
        with graph.as_default():
            ret_nest = graph.capture_call_time_value(closure, spec, key=key, default_value=default_value)
    else:
        ret_nest = closure()
    ret_nest = spec._cast(ret_nest, trace_type.InternalCastContext)
    return spec._to_tensors(ret_nest)