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
@tf_export('xla_rng_bit_generator')
def xla_rng_bit_generator(algorithm: _atypes.TensorFuzzingAnnotation[_atypes.Int32], initial_state: _atypes.TensorFuzzingAnnotation[_atypes.UInt64], shape: _atypes.TensorFuzzingAnnotation[TV_XlaRngBitGenerator_Tshape], dtype: TV_XlaRngBitGenerator_dtype=_dtypes.uint64, name=None):
    """Stateless PRNG bit generator.

  Wraps the XLA RngBitGenerator operator, documented at
   https://www.tensorflow.org/performance/xla/operation_semantics#rngbitgenerator.

  Args:
    algorithm: A `Tensor` of type `int32`. The PRNG algorithm to use, one of
      tf.random.Algorithm.{PHILOX, THREEFRY, AUTO_SELECT}.
    initial_state: A `Tensor` of type `uint64`.
      Initial state for the PRNG algorithm. For THREEFRY, it should be
      a u64[2] and for PHILOX a u64[3].
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The output shape of the generated data.
    dtype: An optional `tf.DType` from: `tf.int32, tf.int64, tf.uint32, tf.uint64`. Defaults to `tf.uint64`.
      The type of the tensor.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_key, output).

    output_key: A `Tensor` of type `uint64`.
    output: A `Tensor` of type `dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaRngBitGenerator', name, algorithm, initial_state, shape, 'dtype', dtype)
            _result = _XlaRngBitGeneratorOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_rng_bit_generator((algorithm, initial_state, shape, dtype, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_rng_bit_generator_eager_fallback(algorithm, initial_state, shape, dtype=dtype, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_rng_bit_generator, (), dict(algorithm=algorithm, initial_state=initial_state, shape=shape, dtype=dtype, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_rng_bit_generator((algorithm, initial_state, shape, dtype, name), None)
        if _result is not NotImplemented:
            return _result
    if dtype is None:
        dtype = _dtypes.uint64
    dtype = _execute.make_type(dtype, 'dtype')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaRngBitGenerator', algorithm=algorithm, initial_state=initial_state, shape=shape, dtype=dtype, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_rng_bit_generator, (), dict(algorithm=algorithm, initial_state=initial_state, shape=shape, dtype=dtype, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'Tshape', _op._get_attr_type('Tshape'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaRngBitGenerator', _inputs_flat, _attrs, _result)
    _result = _XlaRngBitGeneratorOutput._make(_result)
    return _result