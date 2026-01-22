from . import _gi
from ._constants import \
def _type_from_python(self, type_):
    if type_ in self._type_from_pytype_lookup:
        return self._type_from_pytype_lookup[type_]
    elif isinstance(type_, type) and issubclass(type_, (_gi.GObject, _gi.GEnum, _gi.GFlags, _gi.GBoxed, _gi.GInterface)):
        return type_.__gtype__
    elif type_ in (TYPE_NONE, TYPE_INTERFACE, TYPE_CHAR, TYPE_UCHAR, TYPE_INT, TYPE_UINT, TYPE_BOOLEAN, TYPE_LONG, TYPE_ULONG, TYPE_INT64, TYPE_UINT64, TYPE_FLOAT, TYPE_DOUBLE, TYPE_POINTER, TYPE_BOXED, TYPE_PARAM, TYPE_OBJECT, TYPE_STRING, TYPE_PYOBJECT, TYPE_GTYPE, TYPE_STRV, TYPE_VARIANT):
        return type_
    else:
        raise TypeError('Unsupported type: %r' % (type_,))