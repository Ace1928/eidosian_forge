import numbers
from collections import namedtuple
from contextlib import contextmanager
from .bindings import libnvpair
from .ctypes import _type_to_suffix
def _dict_to_nvlist(props, nvlist):
    for k, v in props.items():
        if not isinstance(k, bytes):
            raise TypeError('Unsupported key type ' + type(k).__name__)
        ret = 0
        if isinstance(v, dict):
            ret = _lib.nvlist_add_nvlist(nvlist, k, nvlist_in(v))
        elif isinstance(v, list):
            _nvlist_add_array(nvlist, k, v)
        elif isinstance(v, bytes):
            ret = _lib.nvlist_add_string(nvlist, k, v)
        elif isinstance(v, bool):
            ret = _lib.nvlist_add_boolean_value(nvlist, k, v)
        elif v is None:
            ret = _lib.nvlist_add_boolean(nvlist, k)
        elif isinstance(v, numbers.Integral):
            suffix = _prop_name_to_type_str.get(k, 'uint64')
            cfunc = getattr(_lib, 'nvlist_add_%s' % (suffix,))
            ret = cfunc(nvlist, k, v)
        elif isinstance(v, _ffi.CData) and _ffi.typeof(v) in _type_to_suffix:
            suffix = _type_to_suffix[_ffi.typeof(v)][False]
            cfunc = getattr(_lib, 'nvlist_add_%s' % (suffix,))
            ret = cfunc(nvlist, k, v)
        else:
            raise TypeError('Unsupported value type ' + type(v).__name__)
        if ret != 0:
            raise MemoryError('nvlist_add failed')