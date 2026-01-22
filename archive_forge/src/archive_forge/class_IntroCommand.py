from __future__ import annotations
from contextlib import redirect_stdout
import collections
import dataclasses
import json
import os
from pathlib import Path, PurePath
import sys
import typing as T
from . import build, mesonlib, coredata as cdata
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstJSONPrinter
from .backend import backends
from .dependencies import Dependency
from . import environment
from .interpreterbase import ObjectHolder
from .mesonlib import OptionKey
from .mparser import FunctionNode, ArrayNode, ArgumentNode, BaseStringNode
class IntroCommand:

    def __init__(self, desc: str, func: T.Optional[T.Callable[[], T.Union[dict, list]]]=None, no_bd: T.Optional[T.Callable[[IntrospectionInterpreter], T.Union[dict, list]]]=None) -> None:
        self.desc = desc + '.'
        self.func = func
        self.no_bd = no_bd