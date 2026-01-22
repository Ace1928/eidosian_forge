import ctypes, ctypes.util, operator, sys
from . import model
def _new_struct_or_union(self, kind, name, base_ctypes_class):

    class struct_or_union(base_ctypes_class):
        pass
    struct_or_union.__name__ = '%s_%s' % (kind, name)
    kind1 = kind

    class CTypesStructOrUnion(CTypesBaseStructOrUnion):
        __slots__ = ['_blob']
        _ctype = struct_or_union
        _reftypename = '%s &' % (name,)
        _kind = kind = kind1
    CTypesStructOrUnion._fix_class()
    return CTypesStructOrUnion