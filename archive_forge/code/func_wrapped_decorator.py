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
@functools.wraps(decorator)
def wrapped_decorator(*args, **kwds):
    func = decorator(*args, **kwds)
    func = no_type_check(func)
    return func