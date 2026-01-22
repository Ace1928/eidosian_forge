import numbers
from collections import namedtuple
from contextlib import contextmanager
from .bindings import libnvpair
from .ctypes import _type_to_suffix
def _nvlist_add_array(nvlist, key, array):

    def _is_integer(x):
        return isinstance(x, numbers.Integral) and (not isinstance(x, bool))
    ret = 0
    specimen = array[0]
    is_integer = _is_integer(specimen)
    specimen_ctype = None
    if isinstance(specimen, _ffi.CData):
        specimen_ctype = _ffi.typeof(specimen)
    for element in array[1:]:
        if is_integer and _is_integer(element):
            pass
        elif type(element) is not type(specimen):
            raise TypeError('Array has elements of different types: ' + type(specimen).__name__ + ' and ' + type(element).__name__)
        elif specimen_ctype is not None:
            ctype = _ffi.typeof(element)
            if ctype is not specimen_ctype:
                raise TypeError('Array has elements of different C types: ' + _ffi.typeof(specimen).cname + ' and ' + _ffi.typeof(element).cname)
    if isinstance(specimen, dict):
        c_array = []
        for dictionary in array:
            nvlistp = _ffi.new('nvlist_t **')
            res = _lib.nvlist_alloc(nvlistp, 1, 0)
            if res != 0:
                raise MemoryError('nvlist_alloc failed')
            nested_nvlist = _ffi.gc(nvlistp[0], _lib.nvlist_free)
            _dict_to_nvlist(dictionary, nested_nvlist)
            c_array.append(nested_nvlist)
        ret = _lib.nvlist_add_nvlist_array(nvlist, key, c_array, len(c_array))
    elif isinstance(specimen, bytes):
        c_array = []
        for string in array:
            c_array.append(_ffi.new('char[]', string))
        ret = _lib.nvlist_add_string_array(nvlist, key, c_array, len(c_array))
    elif isinstance(specimen, bool):
        ret = _lib.nvlist_add_boolean_array(nvlist, key, array, len(array))
    elif isinstance(specimen, numbers.Integral):
        suffix = _prop_name_to_type_str.get(key, 'uint64')
        cfunc = getattr(_lib, 'nvlist_add_%s_array' % (suffix,))
        ret = cfunc(nvlist, key, array, len(array))
    elif isinstance(specimen, _ffi.CData) and _ffi.typeof(specimen) in _type_to_suffix:
        suffix = _type_to_suffix[_ffi.typeof(specimen)][True]
        cfunc = getattr(_lib, 'nvlist_add_%s_array' % (suffix,))
        ret = cfunc(nvlist, key, array, len(array))
    else:
        raise TypeError('Unsupported value type ' + type(specimen).__name__)
    if ret != 0:
        raise MemoryError('nvlist_add failed, err = %d' % ret)