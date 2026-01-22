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
class _CallableGenericAlias(_NotIterable, _GenericAlias, _root=True):

    def __repr__(self):
        assert self._name == 'Callable'
        args = self.__args__
        if len(args) == 2 and _is_param_expr(args[0]):
            return super().__repr__()
        return f'typing.Callable[[{', '.join([_type_repr(a) for a in args[:-1]])}], {_type_repr(args[-1])}]'

    def __reduce__(self):
        args = self.__args__
        if not (len(args) == 2 and _is_param_expr(args[0])):
            args = (list(args[:-1]), args[-1])
        return (operator.getitem, (Callable, args))