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
def _collect_type_vars(types, typevar_types=None):
    """Collect all type variable contained in types in order of
    first appearance (lexicographic order). For example::

        _collect_type_vars((T, List[S, T])) == (T, S)
    """
    if typevar_types is None:
        typevar_types = typing.TypeVar
    tvars = []
    for t in types:
        if isinstance(t, typevar_types) and t not in tvars and (not _is_unpack(t)):
            tvars.append(t)
        if _should_collect_from_parameters(t):
            tvars.extend([t for t in t.__parameters__ if t not in tvars])
    return tuple(tvars)