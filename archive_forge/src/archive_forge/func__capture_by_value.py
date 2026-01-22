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
def _capture_by_value(self, op_type, inputs, dtypes, input_types=None, name=None, attrs=None, op_def=None, compute_device=True):
    reverse_captures = dict(((id(v), k) for k, v in self.captures))
    uncaptured_inputs = [reverse_captures.get(id(t), t) for t in inputs]
    with ops.init_scope():
        if context.executing_eagerly():
            attr_list = ('dtype', int(attrs['dtype'].type))
            value, = execute.execute(compat.as_bytes(op_type), 1, uncaptured_inputs, attr_list, context.context())
        else:
            op = ops.get_default_graph()._create_op_internal(op_type, uncaptured_inputs, dtypes, input_types, name, attrs, op_def, compute_device)
            value = op.outputs[0]
    captured_value = self.capture(value)
    return captured_value.op