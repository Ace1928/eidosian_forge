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
def anonymous_multi_device_iterator_eager_fallback(devices, output_types, output_shapes, name, ctx):
    if not isinstance(devices, (list, tuple)):
        raise TypeError("Expected list for 'devices' argument to 'anonymous_multi_device_iterator' Op, not %r." % devices)
    devices = [_execute.make_str(_s, 'devices') for _s in devices]
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'anonymous_multi_device_iterator' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'anonymous_multi_device_iterator' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    _inputs_flat = []
    _attrs = ('devices', devices, 'output_types', output_types, 'output_shapes', output_shapes)
    _result = _execute.execute(b'AnonymousMultiDeviceIterator', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('AnonymousMultiDeviceIterator', _inputs_flat, _attrs, _result)
    _result = _AnonymousMultiDeviceIteratorOutput._make(_result)
    return _result