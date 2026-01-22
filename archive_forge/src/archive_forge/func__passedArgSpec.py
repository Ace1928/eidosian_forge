from __future__ import annotations
import inspect
import sys
from dis import findlinestarts
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from warnings import warn, warn_explicit
from incremental import Version, getVersionString
from typing_extensions import ParamSpec
def _passedArgSpec(argspec, positional, keyword):
    """
    Take an I{inspect.ArgSpec}, a tuple of positional arguments, and a dict of
    keyword arguments, and return a mapping of arguments that were actually
    passed to their passed values.

    @param argspec: The argument specification for the function to inspect.
    @type argspec: I{inspect.ArgSpec}

    @param positional: The positional arguments that were passed.
    @type positional: L{tuple}

    @param keyword: The keyword arguments that were passed.
    @type keyword: L{dict}

    @return: A dictionary mapping argument names (those declared in C{argspec})
        to values that were passed explicitly by the user.
    @rtype: L{dict} mapping L{str} to L{object}
    """
    result: Dict[str, object] = {}
    unpassed = len(argspec.args) - len(positional)
    if argspec.keywords is not None:
        kwargs = result[argspec.keywords] = {}
    if unpassed < 0:
        if argspec.varargs is None:
            raise TypeError('Too many arguments.')
        else:
            result[argspec.varargs] = positional[len(argspec.args):]
    for name, value in zip(argspec.args, positional):
        result[name] = value
    for name, value in keyword.items():
        if name in argspec.args:
            if name in result:
                raise TypeError('Already passed.')
            result[name] = value
        elif argspec.keywords is not None:
            kwargs[name] = value
        else:
            raise TypeError('no such param')
    return result