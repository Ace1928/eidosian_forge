import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
class _TargetInfo(object):

    def __init__(self, except_end_instruction, jump_if_not_exc_instruction=None):
        self.except_end_instruction = except_end_instruction
        self.jump_if_not_exc_instruction = jump_if_not_exc_instruction

    def __str__(self):
        msg = ['_TargetInfo(']
        msg.append(self.except_end_instruction.opname)
        if self.jump_if_not_exc_instruction:
            msg.append(' - ')
            msg.append(self.jump_if_not_exc_instruction.opname)
            msg.append('(')
            msg.append(str(self.jump_if_not_exc_instruction.argval))
            msg.append(')')
        msg.append(')')
        return ''.join(msg)