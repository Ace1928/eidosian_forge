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
def tpu_partitioned_call_eager_fallback(args, device_ordinal: _atypes.TensorFuzzingAnnotation[_atypes.Int32], Tout, f, autotuner_thresh: int, name, ctx):
    if not isinstance(Tout, (list, tuple)):
        raise TypeError("Expected list for 'Tout' argument to 'tpu_partitioned_call' Op, not %r." % Tout)
    Tout = [_execute.make_type(_t, 'Tout') for _t in Tout]
    if autotuner_thresh is None:
        autotuner_thresh = 0
    autotuner_thresh = _execute.make_int(autotuner_thresh, 'autotuner_thresh')
    _attr_Tin, args = _execute.convert_to_mixed_eager_tensors(args, ctx)
    device_ordinal = _ops.convert_to_tensor(device_ordinal, _dtypes.int32)
    _inputs_flat = list(args) + [device_ordinal]
    _attrs = ('Tin', _attr_Tin, 'Tout', Tout, 'f', f, 'autotuner_thresh', autotuner_thresh)
    _result = _execute.execute(b'TPUPartitionedCall', len(Tout), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TPUPartitionedCall', _inputs_flat, _attrs, _result)
    return _result