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
@_SpecialForm
def Optional(self, parameters):
    """Optional[X] is equivalent to Union[X, None]."""
    arg = _type_check(parameters, f'{self} requires a single type.')
    return Union[arg, type(None)]