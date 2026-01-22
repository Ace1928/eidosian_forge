import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
def _decorate_jump_target(self, instruction, instruction_repr):
    if instruction.is_jump_target:
        return ('|', str(instruction.offset), '|', instruction_repr)
    return instruction_repr