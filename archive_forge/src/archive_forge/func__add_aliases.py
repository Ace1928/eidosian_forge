from numpy.compat import unicode
from numpy.core._string_helpers import english_lower
from numpy.core.multiarray import typeinfo, dtype
from numpy.core._dtype import _kind_name
def _add_aliases():
    for name, info in _concrete_typeinfo.items():
        if name in _int_ctypes or name in _uint_ctypes:
            continue
        base, bit, char = bitname(info.type)
        myname = '%s%d' % (base, bit)
        if name in ('longdouble', 'clongdouble') and myname in allTypes:
            continue
        if bit != 0 and base != 'bool':
            allTypes[myname] = info.type
        sctypeDict[char] = info.type
        sctypeDict[myname] = info.type