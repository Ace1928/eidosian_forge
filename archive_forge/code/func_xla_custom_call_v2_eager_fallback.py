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
def xla_custom_call_v2_eager_fallback(operands, call_target_name: str, backend_config: str, has_side_effect: bool, result_dtypes, result_shapes, name, ctx):
    call_target_name = _execute.make_str(call_target_name, 'call_target_name')
    backend_config = _execute.make_str(backend_config, 'backend_config')
    has_side_effect = _execute.make_bool(has_side_effect, 'has_side_effect')
    if not isinstance(result_dtypes, (list, tuple)):
        raise TypeError("Expected list for 'result_dtypes' argument to 'xla_custom_call_v2' Op, not %r." % result_dtypes)
    result_dtypes = [_execute.make_type(_t, 'result_dtypes') for _t in result_dtypes]
    if not isinstance(result_shapes, (list, tuple)):
        raise TypeError("Expected list for 'result_shapes' argument to 'xla_custom_call_v2' Op, not %r." % result_shapes)
    result_shapes = [_execute.make_shape(_s, 'result_shapes') for _s in result_shapes]
    _attr_operand_dtypes, operands = _execute.convert_to_mixed_eager_tensors(operands, ctx)
    _inputs_flat = list(operands)
    _attrs = ('call_target_name', call_target_name, 'backend_config', backend_config, 'has_side_effect', has_side_effect, 'operand_dtypes', _attr_operand_dtypes, 'result_dtypes', result_dtypes, 'result_shapes', result_shapes)
    _result = _execute.execute(b'XlaCustomCallV2', len(result_dtypes), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaCustomCallV2', _inputs_flat, _attrs, _result)
    return _result