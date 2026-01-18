from __future__ import annotations
from .. import mesonlib, mlog
from .disabler import Disabler
from .exceptions import InterpreterException, InvalidArguments
from ._unholder import _unholder
from dataclasses import dataclass
from functools import wraps
import abc
import itertools
import copy
import typing as T
def unholder_return(f: TV_func) -> T.Callable[..., TYPE_var]:

    @wraps(f)
    def wrapped(*wrapped_args: T.Any, **wrapped_kwargs: T.Any) -> T.Any:
        res = f(*wrapped_args, **wrapped_kwargs)
        return _unholder(res)
    return T.cast('T.Callable[..., TYPE_var]', wrapped)