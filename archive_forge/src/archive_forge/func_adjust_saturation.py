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
def adjust_saturation(images: _atypes.TensorFuzzingAnnotation[TV_AdjustSaturation_T], scale: _atypes.TensorFuzzingAnnotation[_atypes.Float32], name=None) -> _atypes.TensorFuzzingAnnotation[TV_AdjustSaturation_T]:
    """Adjust the saturation of one or more images.

  `images` is a tensor of at least 3 dimensions.  The last dimension is
  interpreted as channels, and must be three.

  The input image is considered in the RGB colorspace. Conceptually, the RGB
  colors are first mapped into HSV. A scale is then applied all the saturation
  values, and then remapped back to RGB colorspace.

  Args:
    images: A `Tensor`. Must be one of the following types: `half`, `float32`.
      Images to adjust.  At least 3-D.
    scale: A `Tensor` of type `float32`.
      A float scale to add to the saturation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'AdjustSaturation', name, images, scale)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return adjust_saturation_eager_fallback(images, scale, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('AdjustSaturation', images=images, scale=scale, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('AdjustSaturation', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result