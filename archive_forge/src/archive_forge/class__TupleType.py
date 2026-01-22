from abc import abstractmethod, ABCMeta
import collections
from collections import defaultdict
import collections.abc
import contextlib
import functools
import operator
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
import warnings
from types import WrapperDescriptorType, MethodWrapperType, MethodDescriptorType, GenericAlias
class _TupleType(_SpecialGenericAlias, _root=True):

    @_tp_cache
    def __getitem__(self, params):
        if not isinstance(params, tuple):
            params = (params,)
        if len(params) >= 2 and params[-1] is ...:
            msg = 'Tuple[t, ...]: t must be a type.'
            params = tuple((_type_check(p, msg) for p in params[:-1]))
            return self.copy_with((*params, _TypingEllipsis))
        msg = 'Tuple[t0, t1, ...]: each t must be a type.'
        params = tuple((_type_check(p, msg) for p in params))
        return self.copy_with(params)