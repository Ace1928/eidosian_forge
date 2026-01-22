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
@classmethod
def _OverloadOperator(cls, operator):
    """Defer an operator overload to `tensor_lib.Tensor`.

    We pull the operator out of tensor_lib.Tensor dynamically to avoid ordering
    issues.

    Args:
      operator: string. The operator name.
    """
    if operator == '__eq__' or operator == '__ne__':
        return
    tensor_oper = getattr(tensor_lib.Tensor, operator)

    def _run_op(a, *args, **kwargs):
        return tensor_oper(a.value(), *args, **kwargs)
    functools.update_wrapper(_run_op, tensor_oper)
    setattr(cls, operator, _run_op)