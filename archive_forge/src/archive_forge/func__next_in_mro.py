import abc
from abc import abstractmethod, abstractproperty
import collections
import contextlib
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
def _next_in_mro(cls):
    """Helper for Generic.__new__.

    Returns the class after the last occurrence of Generic or
    Generic[...] in cls.__mro__.
    """
    next_in_mro = object
    for i, c in enumerate(cls.__mro__[:-1]):
        if isinstance(c, GenericMeta) and c._gorg is Generic:
            next_in_mro = cls.__mro__[i + 1]
    return next_in_mro