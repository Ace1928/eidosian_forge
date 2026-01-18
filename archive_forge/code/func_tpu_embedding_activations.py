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
def tpu_embedding_activations(embedding_variable: _atypes.TensorFuzzingAnnotation[_atypes.Float32], sliced_activations: _atypes.TensorFuzzingAnnotation[_atypes.Float32], table_id: int, lookup_id: int, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """An op enabling differentiation of TPU Embeddings.

  This op simply returns its first input, which is assumed to have been sliced
  from the Tensors returned by TPUEmbeddingDequeueActivations. The presence of
  this op, and its first argument being a trainable Variable, enables automatic
  differentiation of graphs containing embeddings via the TPU Embedding Python
  libraries.

  Args:
    embedding_variable: A `Tensor` of type `float32`.
      A trainable variable, enabling optimizers to find this op.
    sliced_activations: A `Tensor` of type `float32`.
      The embedding activations Tensor to return.
    table_id: An `int` that is `>= 0`.
      The id of the table in the embedding layer configuration from which
      these activations were computed.
    lookup_id: An `int` that is `>= 0`.
      Identifier of the set of embedding indices which produced these
      activations.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TPUEmbeddingActivations', name, embedding_variable, sliced_activations, 'table_id', table_id, 'lookup_id', lookup_id)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tpu_embedding_activations_eager_fallback(embedding_variable, sliced_activations, table_id=table_id, lookup_id=lookup_id, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    table_id = _execute.make_int(table_id, 'table_id')
    lookup_id = _execute.make_int(lookup_id, 'lookup_id')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TPUEmbeddingActivations', embedding_variable=embedding_variable, sliced_activations=sliced_activations, table_id=table_id, lookup_id=lookup_id, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('table_id', _op._get_attr_int('table_id'), 'lookup_id', _op._get_attr_int('lookup_id'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TPUEmbeddingActivations', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result