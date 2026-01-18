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
def tpu_replicated_output(input: _atypes.TensorFuzzingAnnotation[TV_TPUReplicatedOutput_T], num_replicas: int, name=None):
    """Connects N outputs from an N-way replicated TPU computation.

  This operation holds a replicated output from a `tpu.replicate()` computation subgraph.
  Each replicated output has the same shape and type alongside the input.

  For example:
  ```
  %computation = "tf.Computation"()
  %replicated_output:2 = "tf.TPUReplicatedOutput"(%computation)
  ```
  The above computation has a replicated output of two replicas.

  Args:
    input: A `Tensor`.
    num_replicas: An `int` that is `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A list of `num_replicas` `Tensor` objects with the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TPUReplicatedOutput', name, input, 'num_replicas', num_replicas)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tpu_replicated_output_eager_fallback(input, num_replicas=num_replicas, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    num_replicas = _execute.make_int(num_replicas, 'num_replicas')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TPUReplicatedOutput', input=input, num_replicas=num_replicas, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('num_replicas', _op._get_attr_int('num_replicas'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TPUReplicatedOutput', _inputs_flat, _attrs, _result)
    return _result