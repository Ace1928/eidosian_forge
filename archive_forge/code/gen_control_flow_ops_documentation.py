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
Forwards `data` to the output port determined by `pred`.

  If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
  the data goes to `output_false`.

  See also `RefSwitch` and `Merge`.

  Args:
    data: A `Tensor`. The tensor to be forwarded to the appropriate output.
    pred: A `Tensor` of type `bool`.
      A scalar that specifies which output port will receive data.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_false, output_true).

    output_false: A `Tensor`. Has the same type as `data`.
    output_true: A `Tensor`. Has the same type as `data`.
  