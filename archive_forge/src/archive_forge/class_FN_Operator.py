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
class FN_Operator(Protocol[_TV_IntegerObject, _TV_ARG1]):

    def __call__(s, self: _TV_IntegerObject, other: _TV_ARG1) -> TYPE_var:
        ...