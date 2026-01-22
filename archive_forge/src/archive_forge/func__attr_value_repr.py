from ipywidgets import register, widget_serialization
from traitlets import validate, TraitError, Undefined
from ipydatawidgets import NDArrayWidget, get_union_array
from .Geometry import _make_key_filter
from .BufferGeometry_autogen import BufferGeometry as BufferGeometryBase
def _attr_value_repr(v):
    try:
        array = get_union_array(v.array)
    except AttributeError:
        from .InterleavedBufferAttribute_autogen import InterleavedBufferAttribute
        if not isinstance(v, InterleavedBufferAttribute):
            raise
        return repr(v)
    if array.size < 50:
        return repr(v)
    return '<%s shape=%r, dtype=%s>' % (v.__class__.__name__, array.shape, array.dtype)