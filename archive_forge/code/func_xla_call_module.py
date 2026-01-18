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
@tf_export('xla_call_module')
def xla_call_module(args, version: int, module: str, Sout, Tout, dim_args_spec=[], platforms=[], function_list=[], has_token_input_output: bool=False, disabled_checks=[], name=None):
    """Invokes a StableHLO module.

  This op is used with JAX native serialization in a TensorFlow context with
  stability guarantees.

  Args:
    args: A list of `Tensor` objects.
      A list of `Tensor` with possibly different types to be passed as arguments
      to the `module`. These are the actual arguments and do not include the
      platform argument (see `platforms`) nor the dimension arguments (see
      `dim_args_spec`).
    version: An `int`.
      Tracks changes the semantics of the op, to support backwards
      compatibility. Minimum supported version is 2. From
      version 2, the op carries a StableHLO text or bytecode `module`. From
      version 3, the op also supports the `platforms` attribute. From version 4,
      the op carries a StableHLO module with compatibility guarantees. From version
      5, XLACallModule can include `stablehlo.custom_call` op to execute tf
      functions. From version 6 the op supports the `disabled_checks` attribute.
      See more versioning details at https://github.com/search?q=repo%3Atensorflow%2Ftensorflow+path%3Axla_call_module+%22int+VERSION_MAXIMUM_SUPPORTED%22&type=code.
    module: A `string`.
      A serialized computation, a text or bytecode representation of
      an mlir.Module. The return type must be a tuple if and only if the `Sout` is
      a list with 0 or more than 1 elements. The length of `Tout` and
      `Sout` must match. This op always returns a tuple of results, even if the
      module returns a single result.
    Sout: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      List of output tensor shapes.
    Tout: A list of `tf.DTypes`. List of output tensor data types.
    dim_args_spec: An optional list of `strings`. Defaults to `[]`.
      in presence of dynamic shapes, this is the specification for the
      dimension arguments. In absence of dynamic shapes this list is empty. The
      `module` takes one 0-dimensional integer tensor dimension argument for each
      element of `dim_spec_args`. The dimension arguments come after the platform
      index argument and before the actual arguments. Each specification is a
      string of the form "<arg_idx>.<axis_idx>" that specifies that the value of
      the corresponding dimension argument must be "args[arg_idx].shape[axis_idx]",
      where "args" are the actual array arguments.
      This attribute is not used anymore in modules serialized with version 5
      after March 28th, 2023 and JAX OSS versions higher than 0.4.6.
      TODO(b/283439649): remove support for dim_args_spec.
    platforms: An optional list of `strings`. Defaults to `[]`.
      the list of platforms supported by `module`. The list can contain
      the strings "CPU", "CUDA", "ROCM", or "TPU". It is an error to compile
      this op for a platform that does not appear in the list. This check can be
      disabled using `disabled_checks`. If the list contains more than
      one platform, then the `module` takes one additional 0-dimensional
      integer-tensor parameter in the first position, encoding the index in
      `platforms` of the current compilation platform. This parameter has value 0
      if the plaform is not among `platforms` and the check has been disabled.
      The list can be empty in old versions (earlier than 6) to denote that no
      platform checking must be performed at loading time.
    function_list: An optional list of functions decorated with @Defun. Defaults to `[]`.
      This list contains the TensorFlow FunctionDefs that are used by
      the XLACallModule. If the XLACallModule contains `stablehlo.custom_call`
      operations, they can call TensorFlow graph functions outside of the
      XLACallModule. This `function_list` attribute registers the dependency of the
      XLACallModule on those functions. This attribute was added in version 5.
    has_token_input_output: An optional `bool`. Defaults to `False`.
      If true, the embedded StableHLO module's main function
      must take a `!stablehlo.token` as its first argument and returns a token as
      its first result. This can be used in conjunction with the TF2XLA's side
      effect mechanism in order to model side effects.
    disabled_checks: An optional list of `strings`. Defaults to `[]`.
      A list of strings describing the safety checks that were
      disabled at serialization time. This attribute was added in version 6.
      For more details see
      https://github.com/search?q=repo%3Agoogle%2Fjax+path%3Ajax_export+%22class+DisabledSafetyCheck%22&type=code.
      This list, supplemented with a comma-separate list of directives specified
      using the flag --tf_xla_call_module_disabled_checks,
      is used at module loading time to skip the corresponding checks.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaCallModule', name, args, 'version', version, 'module', module, 'Sout', Sout, 'Tout', Tout, 'dim_args_spec', dim_args_spec, 'platforms', platforms, 'function_list', function_list, 'has_token_input_output', has_token_input_output, 'disabled_checks', disabled_checks)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_call_module((args, version, module, Sout, Tout, dim_args_spec, platforms, function_list, has_token_input_output, disabled_checks, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_call_module_eager_fallback(args, version=version, module=module, Sout=Sout, Tout=Tout, dim_args_spec=dim_args_spec, platforms=platforms, function_list=function_list, has_token_input_output=has_token_input_output, disabled_checks=disabled_checks, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_call_module, (), dict(args=args, version=version, module=module, Sout=Sout, Tout=Tout, dim_args_spec=dim_args_spec, platforms=platforms, function_list=function_list, has_token_input_output=has_token_input_output, disabled_checks=disabled_checks, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_call_module((args, version, module, Sout, Tout, dim_args_spec, platforms, function_list, has_token_input_output, disabled_checks, name), None)
        if _result is not NotImplemented:
            return _result
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
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaCallModule', args=args, version=version, module=module, Sout=Sout, Tout=Tout, dim_args_spec=dim_args_spec, platforms=platforms, function_list=function_list, has_token_input_output=has_token_input_output, disabled_checks=disabled_checks, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_call_module, (), dict(args=args, version=version, module=module, Sout=Sout, Tout=Tout, dim_args_spec=dim_args_spec, platforms=platforms, function_list=function_list, has_token_input_output=has_token_input_output, disabled_checks=disabled_checks, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if not _result:
        return _op
    if _execute.must_record_gradient():
        _attrs = ('version', _op._get_attr_int('version'), 'module', _op.get_attr('module'), 'Sout', _op.get_attr('Sout'), 'Tout', _op.get_attr('Tout'), 'Tin', _op.get_attr('Tin'), 'dim_args_spec', _op.get_attr('dim_args_spec'), 'platforms', _op.get_attr('platforms'), 'function_list', _op.get_attr('function_list'), 'has_token_input_output', _op._get_attr_bool('has_token_input_output'), 'disabled_checks', _op.get_attr('disabled_checks'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaCallModule', _inputs_flat, _attrs, _result)
    return _result