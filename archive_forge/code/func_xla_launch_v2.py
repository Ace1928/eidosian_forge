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
@tf_export('xla_launch_v2')
def xla_launch_v2(args, Tresults, constants, resources, function, name=None):
    """XLA Launch Op. For use by the XLA JIT only.

  Args:
    args: A list of `Tensor` objects.
    Tresults: A list of `tf.DTypes`.
    constants: A list of `ints`.
    resources: A list of `ints`.
    function: A function decorated with @Defun.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tresults`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaLaunchV2', name, args, 'Tresults', Tresults, 'constants', constants, 'resources', resources, 'function', function)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_launch_v2((args, Tresults, constants, resources, function, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_launch_v2_eager_fallback(args, Tresults=Tresults, constants=constants, resources=resources, function=function, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_launch_v2, (), dict(args=args, Tresults=Tresults, constants=constants, resources=resources, function=function, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_launch_v2((args, Tresults, constants, resources, function, name), None)
        if _result is not NotImplemented:
            return _result
    if not isinstance(Tresults, (list, tuple)):
        raise TypeError("Expected list for 'Tresults' argument to 'xla_launch_v2' Op, not %r." % Tresults)
    Tresults = [_execute.make_type(_t, 'Tresults') for _t in Tresults]
    if not isinstance(constants, (list, tuple)):
        raise TypeError("Expected list for 'constants' argument to 'xla_launch_v2' Op, not %r." % constants)
    constants = [_execute.make_int(_i, 'constants') for _i in constants]
    if not isinstance(resources, (list, tuple)):
        raise TypeError("Expected list for 'resources' argument to 'xla_launch_v2' Op, not %r." % resources)
    resources = [_execute.make_int(_i, 'resources') for _i in resources]
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaLaunchV2', args=args, Tresults=Tresults, constants=constants, resources=resources, function=function, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_launch_v2, (), dict(args=args, Tresults=Tresults, constants=constants, resources=resources, function=function, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Targs', _op.get_attr('Targs'), 'Tresults', _op.get_attr('Tresults'), 'constants', _op.get_attr('constants'), 'resources', _op.get_attr('resources'), 'function', _op.get_attr('function'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaLaunchV2', _inputs_flat, _attrs, _result)
    return _result