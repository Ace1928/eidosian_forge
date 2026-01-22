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
@_LiteralSpecialForm
@_tp_cache(typed=True)
def Literal(self, *parameters):
    """Special typing form to define literal types (a.k.a. value types).

    This form can be used to indicate to type checkers that the corresponding
    variable or function parameter has a value equivalent to the provided
    literal (or one of several literals)::

        def validate_simple(data: Any) -> Literal[True]:  # always returns True
            ...

        MODE = Literal['r', 'rb', 'w', 'wb']
        def open_helper(file: str, mode: MODE) -> str:
            ...

        open_helper('/some/path', 'r')  # Passes type check
        open_helper('/other/path', 'typo')  # Error in type checker

    Literal[...] cannot be subclassed. At runtime, an arbitrary value
    is allowed as type argument to Literal[...], but type checkers may
    impose restrictions.
    """
    parameters = _flatten_literal_params(parameters)
    try:
        parameters = tuple((p for p, _ in _deduplicate(list(_value_and_type_iter(parameters)))))
    except TypeError:
        pass
    return _LiteralGenericAlias(self, parameters)