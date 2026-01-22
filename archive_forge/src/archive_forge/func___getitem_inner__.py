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
@_tp_cache
def __getitem_inner__(self, params):
    args, result = params
    msg = 'Callable[args, result]: result must be a type.'
    result = _type_check(result, msg)
    if args is Ellipsis:
        return self.copy_with((_TypingEllipsis, result))
    if not isinstance(args, tuple):
        args = (args,)
    args = tuple((_type_convert(arg) for arg in args))
    params = args + (result,)
    return self.copy_with(params)