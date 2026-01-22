import abc
from abc import abstractmethod, abstractproperty
import collections
import contextlib
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
def _type_vars(types):
    tvars = []
    _get_type_vars(types, tvars)
    return tuple(tvars)