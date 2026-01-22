import dis
from contextlib import contextmanager
import builtins
import operator
from typing import Iterator
from functools import reduce
from numba.core import (
from numba.core.utils import (
from .rvsdg.bc2rvsdg import (
from .rvsdg.regionpasses import RegionVisitor
def _get_first_bytecode(ops: list[Op]) -> dis.Instruction | None:
    for bc in (op.bc_inst for op in ops if op.bc_inst is not None):
        return bc
    return None