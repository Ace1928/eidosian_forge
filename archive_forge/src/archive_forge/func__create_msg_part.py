import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
def _create_msg_part(self, instruction, tok=None, line=None):
    dec = self._decorate_jump_target
    if line is None or line in (self.BIG_LINE_INT, self.SMALL_LINE_INT):
        line = self.op_offset_to_line[instruction.offset]
    argrepr = instruction.argrepr
    if isinstance(argrepr, str) and argrepr.startswith('NULL + '):
        argrepr = argrepr[7:]
    return _MsgPart(line, tok if tok is not None else dec(instruction, argrepr))