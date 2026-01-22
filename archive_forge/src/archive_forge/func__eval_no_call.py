import ast
import builtins
import dis
import enum
import inspect
import re
import typing
import warnings
from textwrap import dedent
from typing import Type
import torch
from torch._C import (
from torch._sources import get_source_lines_and_file
from .._jit_internal import (  # type: ignore[attr-defined]
from ._state import _get_script_class
from torch._ops import OpOverloadPacket
def _eval_no_call(stmt, glob, loc):
    """Evaluate statement as long as it does not contain any method/function calls."""
    bytecode = compile(stmt, '', mode='eval')
    for insn in dis.get_instructions(bytecode):
        if 'CALL' in insn.opname:
            raise RuntimeError(f"Type annotation should not contain calls, but '{stmt}' does")
    return eval(bytecode, glob, loc)