import inspect
import keyword
import pydoc
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, Dict, List, ContextManager
from types import MemberDescriptorType, TracebackType
from ._typing_compat import Literal
from pygments.token import Token
from pygments.lexers import Python3Lexer
from .lazyre import LazyReCompile
def _fix_default_values(f: Callable, argspec: ArgSpec) -> ArgSpec:
    """Functions taking default arguments that are references to other objects
    will cause breakage, so we swap out the object itself with the name it was
    referenced with in the source by parsing the source itself!"""
    if argspec.defaults is None and argspec.kwonly_defaults is None:
        return argspec
    try:
        src, _ = inspect.getsourcelines(f)
    except (OSError, IndexError):
        return argspec
    except TypeError:
        if argspec.defaults is not None:
            argspec.defaults = [_Repr(str(value)) for value in argspec.defaults]
        if argspec.kwonly_defaults is not None:
            argspec.kwonly_defaults = {key: _Repr(str(value)) for key, value in argspec.kwonly_defaults.items()}
        return argspec
    kwparsed = parsekeywordpairs(''.join(src))
    if argspec.defaults is not None:
        values = list(argspec.defaults)
        keys = argspec.args[-len(values):]
        for i, key in enumerate(keys):
            values[i] = _Repr(kwparsed[key])
        argspec.defaults = values
    if argspec.kwonly_defaults is not None:
        for key in argspec.kwonly_defaults.keys():
            argspec.kwonly_defaults[key] = _Repr(kwparsed[key])
    return argspec