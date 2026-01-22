import numbers
from collections import namedtuple
from contextlib import contextmanager
from .bindings import libnvpair
from .ctypes import _type_to_suffix
def _nvlist_to_dict(nvlist, props):
    pair = _lib.nvlist_next_nvpair(nvlist, _ffi.NULL)
    while pair != _ffi.NULL:
        name = _ffi.string(_lib.nvpair_name(pair))
        typeid = int(_lib.nvpair_type(pair))
        typeinfo = _type_info(typeid)
        is_array = typeinfo.is_array
        cfunc = getattr(_lib, 'nvpair_value_%s' % (typeinfo.suffix,), None)
        val = None
        ret = 0
        if is_array:
            valptr = _ffi.new(typeinfo.ctype)
            lenptr = _ffi.new('uint_t *')
            ret = cfunc(pair, valptr, lenptr)
            if ret != 0:
                raise RuntimeError('nvpair_value failed')
            length = int(lenptr[0])
            val = []
            for i in range(length):
                val.append(typeinfo.convert(valptr[0][i]))
        elif typeid == _lib.DATA_TYPE_BOOLEAN:
            val = None
        else:
            valptr = _ffi.new(typeinfo.ctype)
            ret = cfunc(pair, valptr)
            if ret != 0:
                raise RuntimeError('nvpair_value failed')
            val = typeinfo.convert(valptr[0])
        props[name] = val
        pair = _lib.nvlist_next_nvpair(nvlist, pair)
    return props