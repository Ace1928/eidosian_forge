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
def xla_launch_v2_eager_fallback(args, Tresults, constants, resources, function, name, ctx):
    if not isinstance(Tresults, (list, tuple)):
        raise TypeError("Expected list for 'Tresults' argument to 'xla_launch_v2' Op, not %r." % Tresults)
    Tresults = [_execute.make_type(_t, 'Tresults') for _t in Tresults]
    if not isinstance(constants, (list, tuple)):
        raise TypeError("Expected list for 'constants' argument to 'xla_launch_v2' Op, not %r." % constants)
    constants = [_execute.make_int(_i, 'constants') for _i in constants]
    if not isinstance(resources, (list, tuple)):
        raise TypeError("Expected list for 'resources' argument to 'xla_launch_v2' Op, not %r." % resources)
    resources = [_execute.make_int(_i, 'resources') for _i in resources]
    _attr_Targs, args = _execute.convert_to_mixed_eager_tensors(args, ctx)
    _inputs_flat = list(args)
    _attrs = ('Targs', _attr_Targs, 'Tresults', Tresults, 'constants', constants, 'resources', resources, 'function', function)
    _result = _execute.execute(b'XlaLaunchV2', len(Tresults), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaLaunchV2', _inputs_flat, _attrs, _result)
    return _result