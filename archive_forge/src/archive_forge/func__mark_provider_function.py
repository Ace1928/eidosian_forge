import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
def _mark_provider_function(function: Callable, *, allow_multi: bool) -> None:
    scope_ = getattr(function, '__scope__', None)
    try:
        annotations = get_type_hints(function)
    except NameError:
        return_type = '__deferred__'
    else:
        return_type = annotations['return']
        _validate_provider_return_type(function, cast(type, return_type), allow_multi)
    function.__binding__ = Binding(return_type, inject(function), scope_)