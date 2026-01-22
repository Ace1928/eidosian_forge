from __future__ import print_function, absolute_import
import sys
import re
import warnings
import types
import keyword
import functools
from shibokensupport.signature.mapping import (type_map, update_mapping,
from shibokensupport.signature.lib.tool import (SimpleNamespace,
from inspect import currentframe
def _resolve_value(thing, valtype, line):
    if thing in ('0', 'None') and valtype:
        if valtype.startswith('PySide2.') or valtype.startswith('typing.'):
            return None
        mapped = type_map[valtype]
        name = mapped.__name__ if hasattr(mapped, '__name__') else str(mapped)
        thing = 'zero({})'.format(name)
    if thing in type_map:
        return type_map[thing]
    res = make_good_value(thing, valtype)
    if res is not None:
        type_map[thing] = res
        return res
    res = try_to_guess(thing, valtype) if valtype else None
    if res is not None:
        type_map[thing] = res
        return res
    warnings.warn('pyside_type_init:\n\n        UNRECOGNIZED:   {!r}\n        OFFENDING LINE: {!r}\n        '.format(thing, line), RuntimeWarning)
    return thing