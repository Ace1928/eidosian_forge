import builtins
import inspect
import operator
import warnings
import textwrap
import re
from functools import reduce
import numpy as np
import numpy.core.umath as umath
import numpy.core.numerictypes as ntypes
from numpy.core import multiarray as mu
from numpy import ndarray, amax, amin, iscomplexobj, bool_, _NoValue
from numpy import array as narray
from numpy.lib.function_base import angle
from numpy.compat import (
from numpy import expand_dims
from numpy.core.numeric import normalize_axis_tuple
frombuffer = _convert2ma(
fromfunction = _convert2ma(
def _update_from(self, obj):
    """
        Copies some attributes of obj to self.

        """
    if isinstance(obj, ndarray):
        _baseclass = type(obj)
    else:
        _baseclass = ndarray
    _optinfo = {}
    _optinfo.update(getattr(obj, '_optinfo', {}))
    _optinfo.update(getattr(obj, '_basedict', {}))
    if not isinstance(obj, MaskedArray):
        _optinfo.update(getattr(obj, '__dict__', {}))
    _dict = dict(_fill_value=getattr(obj, '_fill_value', None), _hardmask=getattr(obj, '_hardmask', False), _sharedmask=getattr(obj, '_sharedmask', False), _isfield=getattr(obj, '_isfield', False), _baseclass=getattr(obj, '_baseclass', _baseclass), _optinfo=_optinfo, _basedict=_optinfo)
    self.__dict__.update(_dict)
    self.__dict__.update(_optinfo)
    return