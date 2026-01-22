from __future__ import annotations
from .. import environment, mparser, mesonlib
from .baseobjects import (
from .exceptions import (
from .decorators import FeatureNew
from .disabler import Disabler, is_disabled
from .helpers import default_resolve_key, flatten, resolve_second_level_holders, stringifyUserArguments
from .operator import MesonOperator
from ._unholder import _unholder
import os, copy, re, pathlib
import typing as T
import textwrap
def _is_project(ast: mparser.CodeBlockNode) -> object:
    if not isinstance(ast, mparser.CodeBlockNode):
        raise InvalidCode('AST is of invalid type. Possibly a bug in the parser.')
    if not ast.lines:
        raise InvalidCode('No statements in code.')
    first = ast.lines[0]
    return isinstance(first, mparser.FunctionNode) and first.func_name.value == 'project'