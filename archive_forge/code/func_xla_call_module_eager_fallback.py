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
def xla_call_module_eager_fallback(args, version: int, module: str, Sout, Tout, dim_args_spec, platforms, function_list, has_token_input_output: bool, disabled_checks, name, ctx):
    version = _execute.make_int(version, 'version')
    module = _execute.make_str(module, 'module')
    if not isinstance(Sout, (list, tuple)):
        raise TypeError("Expected list for 'Sout' argument to 'xla_call_module' Op, not %r." % Sout)
    Sout = [_execute.make_shape(_s, 'Sout') for _s in Sout]
    if not isinstance(Tout, (list, tuple)):
        raise TypeError("Expected list for 'Tout' argument to 'xla_call_module' Op, not %r." % Tout)
    Tout = [_execute.make_type(_t, 'Tout') for _t in Tout]
    if dim_args_spec is None:
        dim_args_spec = []
    if not isinstance(dim_args_spec, (list, tuple)):
        raise TypeError("Expected list for 'dim_args_spec' argument to 'xla_call_module' Op, not %r." % dim_args_spec)
    dim_args_spec = [_execute.make_str(_s, 'dim_args_spec') for _s in dim_args_spec]
    if platforms is None:
        platforms = []
    if not isinstance(platforms, (list, tuple)):
        raise TypeError("Expected list for 'platforms' argument to 'xla_call_module' Op, not %r." % platforms)
    platforms = [_execute.make_str(_s, 'platforms') for _s in platforms]
    if function_list is None:
        function_list = []
    if not isinstance(function_list, (list, tuple)):
        raise TypeError("Expected list for 'function_list' argument to 'xla_call_module' Op, not %r." % function_list)
    if has_token_input_output is None:
        has_token_input_output = False
    has_token_input_output = _execute.make_bool(has_token_input_output, 'has_token_input_output')
    if disabled_checks is None:
        disabled_checks = []
    if not isinstance(disabled_checks, (list, tuple)):
        raise TypeError("Expected list for 'disabled_checks' argument to 'xla_call_module' Op, not %r." % disabled_checks)
    disabled_checks = [_execute.make_str(_s, 'disabled_checks') for _s in disabled_checks]
    _attr_Tin, args = _execute.convert_to_mixed_eager_tensors(args, ctx)
    _inputs_flat = list(args)
    _attrs = ('version', version, 'module', module, 'Sout', Sout, 'Tout', Tout, 'Tin', _attr_Tin, 'dim_args_spec', dim_args_spec, 'platforms', platforms, 'function_list', function_list, 'has_token_input_output', has_token_input_output, 'disabled_checks', disabled_checks)
    _result = _execute.execute(b'XlaCallModule', len(Tout), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaCallModule', _inputs_flat, _attrs, _result)
    return _result