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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_key_value_sort')
def xla_key_value_sort(keys: _atypes.TensorFuzzingAnnotation[TV_XlaKeyValueSort_K], values: _atypes.TensorFuzzingAnnotation[TV_XlaKeyValueSort_V], name=None):
    """Wraps the XLA Sort operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#sort
  .

  Sorts a tensor. Currently only sorts in ascending order are supported.

  Args:
    keys: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      A `Tensor` of type K.
    values: A `Tensor`. A `Tensor` of type V.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sorted_keys, sorted_values).

    sorted_keys: A `Tensor`. Has the same type as `keys`. A `Tensor` of type K.
    sorted_values: A `Tensor`. Has the same type as `values`. A `Tensor` of type V.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaKeyValueSort', name, keys, values)
            _result = _XlaKeyValueSortOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_key_value_sort((keys, values, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_key_value_sort_eager_fallback(keys, values, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_key_value_sort, (), dict(keys=keys, values=values, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_key_value_sort((keys, values, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaKeyValueSort', keys=keys, values=values, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_key_value_sort, (), dict(keys=keys, values=values, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('K', _op._get_attr_type('K'), 'V', _op._get_attr_type('V'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaKeyValueSort', _inputs_flat, _attrs, _result)
    _result = _XlaKeyValueSortOutput._make(_result)
    return _result