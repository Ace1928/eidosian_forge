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
def xla_if_eager_fallback(cond: _atypes.TensorFuzzingAnnotation[TV_XlaIf_Tcond], inputs, then_branch, else_branch, Tout, name, ctx):
    if not isinstance(Tout, (list, tuple)):
        raise TypeError("Expected list for 'Tout' argument to 'xla_if' Op, not %r." % Tout)
    Tout = [_execute.make_type(_t, 'Tout') for _t in Tout]
    _attr_Tcond, (cond,) = _execute.args_to_matching_eager([cond], ctx, [])
    _attr_Tin, inputs = _execute.convert_to_mixed_eager_tensors(inputs, ctx)
    _inputs_flat = [cond] + list(inputs)
    _attrs = ('Tcond', _attr_Tcond, 'then_branch', then_branch, 'else_branch', else_branch, 'Tin', _attr_Tin, 'Tout', Tout)
    _result = _execute.execute(b'XlaIf', len(Tout), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaIf', _inputs_flat, _attrs, _result)
    return _result