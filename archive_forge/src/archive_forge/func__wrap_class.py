import logging
import functools
import inspect
import itertools
import sys
import textwrap
import types
from pyomo.common.errors import DeveloperError
def _wrap_class(cls, msg, logger, version, remove_in):
    _doc = None
    for field in ('__new__', '__init__', '__new_member__'):
        _funcDoc = getattr(getattr(cls, field, None), '__doc__', '') or ''
        _flagIdx = _funcDoc.find(_doc_flag)
        if _flagIdx >= 0:
            _doc = _funcDoc[_flagIdx:]
            break
    if msg is not None or _doc is None:
        _doc = _deprecation_docstring(cls, msg, version, remove_in)
    if cls.__doc__:
        _doc = cls.__doc__ + '\n\n' + _doc
    cls.__doc__ = 'DEPRECATED.\n\n' + _doc
    if _flagIdx < 0:
        field = '__init__'
        for c in reversed(cls.__mro__):
            for f in ('__new__', '__init__'):
                if getattr(c, f, None) is not getattr(cls, f, None):
                    field = f
        setattr(cls, field, _wrap_func(getattr(cls, field), msg, logger, version, remove_in))
    return cls