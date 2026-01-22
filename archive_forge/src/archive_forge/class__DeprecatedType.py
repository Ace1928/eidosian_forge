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
class _DeprecatedType(type):

    def __getattribute__(cls, name):
        if name not in ('__dict__', '__module__') and name in cls.__dict__:
            warnings.warn(f'{cls.__name__} is deprecated, import directly from typing instead. {cls.__name__} will be removed in Python 3.12.', DeprecationWarning, stacklevel=2)
        return super().__getattribute__(name)