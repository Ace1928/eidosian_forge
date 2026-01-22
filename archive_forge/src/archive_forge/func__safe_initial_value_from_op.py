import abc
import enum
import functools
import itertools
import os
from tensorflow.core.framework import variable_pb2
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_should_use
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _safe_initial_value_from_op(name, op, op_cache):
    """Replace dependencies on variables with their initialized values.

  Args:
    name: Variable name.
    op: An `Operation`. The operation to replace.
    op_cache: A dict mapping operation names to `Operation`s. Used to memoize
      the results so as to avoid creating redundant operations.

  Returns:
    An `Operation` compatible with `op`. Any inputs that lead to variable
    values will be replaced with a corresponding graph that uses the
    variable's initialized values. This is done on a best-effort basis. If no
    modifications need to be made then `op` will be returned unchanged.
  """
    op_type = op.node_def.op
    if op_type in ('IsVariableInitialized', 'VarIsInitializedOp', 'ReadVariableOp', 'If'):
        return op
    if op_type in ('Variable', 'VariableV2', 'VarHandleOp'):
        initialized_value = _find_initialized_value_for_variable(op)
        return op if initialized_value is None else initialized_value.op
    modified = False
    new_op_inputs = []
    for op_input in op.inputs:
        new_op_input = _safe_initial_value_from_tensor(name, op_input, op_cache)
        new_op_inputs.append(new_op_input)
        modified = modified or new_op_input != op_input
    if modified:
        new_op_type = op_type
        if new_op_type == 'RefSwitch':
            new_op_type = 'Switch'
        new_op_name = op.node_def.name + '_' + name
        new_op_name = new_op_name.replace(':', '_')
        return op.graph.create_op(new_op_type, new_op_inputs, op._output_types, name=new_op_name, attrs=op.node_def.attr)
    return op