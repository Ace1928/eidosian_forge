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
def tpu_partitioned_input_v2(inputs: List[_atypes.TensorFuzzingAnnotation[TV_TPUPartitionedInputV2_T]], partition_dims, is_packed: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_TPUPartitionedInputV2_T]:
    """An op that groups a list of partitioned inputs together. Supports ND sharding.

  Args:
    inputs: A list of at least 1 `Tensor` objects with the same type.
      A list of partitioned inputs which must have the same shape.
    partition_dims: A list of `ints`.
      A list of integers describing how each dimension is partitioned. Emptiness
      indicates the inputs are replicated.
    is_packed: An optional `bool`. Defaults to `False`.
      Indicates whether the input is a packed resource.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TPUPartitionedInputV2', name, inputs, 'partition_dims', partition_dims, 'is_packed', is_packed)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tpu_partitioned_input_v2_eager_fallback(inputs, partition_dims=partition_dims, is_packed=is_packed, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("Expected list for 'inputs' argument to 'tpu_partitioned_input_v2' Op, not %r." % inputs)
    _attr_N = len(inputs)
    if not isinstance(partition_dims, (list, tuple)):
        raise TypeError("Expected list for 'partition_dims' argument to 'tpu_partitioned_input_v2' Op, not %r." % partition_dims)
    partition_dims = [_execute.make_int(_i, 'partition_dims') for _i in partition_dims]
    if is_packed is None:
        is_packed = False
    is_packed = _execute.make_bool(is_packed, 'is_packed')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TPUPartitionedInputV2', inputs=inputs, partition_dims=partition_dims, is_packed=is_packed, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('N', _op._get_attr_int('N'), 'T', _op._get_attr_type('T'), 'partition_dims', _op.get_attr('partition_dims'), 'is_packed', _op._get_attr_bool('is_packed'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TPUPartitionedInputV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result