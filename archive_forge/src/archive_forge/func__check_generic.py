import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
def _check_generic(cls, parameters, elen=_marker):
    """Check correct count for parameters of a generic cls (internal helper).
    This gives a nice error message in case of count mismatch.
    """
    if not elen:
        raise TypeError(f'{cls} is not a generic class')
    if elen is _marker:
        if not hasattr(cls, '__parameters__') or not cls.__parameters__:
            raise TypeError(f'{cls} is not a generic class')
        elen = len(cls.__parameters__)
    alen = len(parameters)
    if alen != elen:
        if hasattr(cls, '__parameters__'):
            parameters = [p for p in cls.__parameters__ if not _is_unpack(p)]
            num_tv_tuples = sum((isinstance(p, TypeVarTuple) for p in parameters))
            if num_tv_tuples > 0 and alen >= elen - num_tv_tuples:
                return
        raise TypeError(f'Too {('many' if alen > elen else 'few')} parameters for {cls}; actual {alen}, expected {elen}')