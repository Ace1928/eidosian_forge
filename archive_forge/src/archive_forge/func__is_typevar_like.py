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
def _is_typevar_like(x: Any) -> bool:
    return isinstance(x, (TypeVar, ParamSpec)) or _is_unpacked_typevartuple(x)