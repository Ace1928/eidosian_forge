import numbers
import warnings
from .multiarray import (
from .._utils import set_module
from ._string_helpers import (
from ._type_aliases import (
from ._dtype import _kind_name
from builtins import bool, int, float, complex, object, str, bytes
from numpy.compat import long, unicode
def _construct_lookups():
    for name, info in _concrete_typeinfo.items():
        obj = info.type
        nbytes[obj] = info.bits // 8
        _alignment[obj] = info.alignment
        if len(info) > 5:
            _maxvals[obj] = info.max
            _minvals[obj] = info.min
        else:
            _maxvals[obj] = None
            _minvals[obj] = None