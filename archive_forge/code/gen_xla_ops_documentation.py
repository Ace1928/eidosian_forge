import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
output = input; While (Cond(output)) { output = Body(output) }

  Args:
    input: A list of `Tensor` objects.
      A list of input tensors whose types are T.
    cond: A function decorated with @Defun.
      A function takes 'input' and returns a tensor.  If the tensor is
      a scalar of non-boolean, the scalar is converted to a boolean
      according to the following rule: if the scalar is a numerical
      value, non-zero means True and zero means False; if the scalar is
      a string, non-empty means True and empty means False. If the
      tensor is not a scalar, non-emptiness means True and False
      otherwise.
    body: A function decorated with @Defun.
      A function that takes a list of tensors and returns another
      list of tensors. Both lists have the same types as specified by T.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `input`.
    A list of output tensors whose types are T.
  