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
@tf_export('xla_variadic_sort')
def xla_variadic_sort(inputs, dimension: _atypes.TensorFuzzingAnnotation[_atypes.Int32], comparator, is_stable: bool, name=None):
    """Wraps the XLA Sort operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#sort
  .

  Sorts one or more tensors, with support for custom comparator, dimension, and
  is_stable attributes.

  Args:
    inputs: A list of `Tensor` objects.
      A list of `Tensor` of identical shape but possibly different types.
    dimension: A `Tensor` of type `int32`.
      The dimension along which to sort. Must be a compile-time constant.
    comparator: A function decorated with @Defun.
      A comparator function to apply to 2*N scalars and returning a
      boolean. N is the number of sort inputs. If you want to sort in ascending
      order then the comparator should perform a less-than comparison.
    is_stable: A `bool`. Whether to use stable sort.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `inputs`.
    A list of `Tensor` of same shape and types as the `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaVariadicSort', name, inputs, dimension, 'comparator', comparator, 'is_stable', is_stable)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_variadic_sort((inputs, dimension, comparator, is_stable, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_variadic_sort_eager_fallback(inputs, dimension, comparator=comparator, is_stable=is_stable, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_variadic_sort, (), dict(inputs=inputs, dimension=dimension, comparator=comparator, is_stable=is_stable, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_variadic_sort((inputs, dimension, comparator, is_stable, name), None)
        if _result is not NotImplemented:
            return _result
    is_stable = _execute.make_bool(is_stable, 'is_stable')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaVariadicSort', inputs=inputs, dimension=dimension, comparator=comparator, is_stable=is_stable, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_variadic_sort, (), dict(inputs=inputs, dimension=dimension, comparator=comparator, is_stable=is_stable, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op.get_attr('T'), 'comparator', _op.get_attr('comparator'), 'is_stable', _op._get_attr_bool('is_stable'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaVariadicSort', _inputs_flat, _attrs, _result)
    return _result