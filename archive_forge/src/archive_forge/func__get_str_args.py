import ast
import collections
import dataclasses
import secrets
import sys
from functools import lru_cache
from importlib.util import find_spec
from typing import Dict, List, Optional, Tuple
from black.output import out
from black.report import NothingChanged
def _get_str_args(args: List[ast.expr]) -> List[str]:
    str_args = []
    for arg in args:
        assert isinstance(arg, ast.Str)
        str_args.append(arg.s)
    return str_args