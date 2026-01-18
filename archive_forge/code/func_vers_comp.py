from __future__ import annotations
from .. import mesonlib
from .. import mlog
from .common import cmake_is_debug
import typing as T
def vers_comp(op: str, arg: str) -> str:
    col_pos = arg.find(',')
    if col_pos < 0:
        return '0'
    else:
        return '1' if mesonlib.version_compare(arg[:col_pos], '{}{}'.format(op, arg[col_pos + 1:])) else '0'