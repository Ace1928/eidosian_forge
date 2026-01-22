import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
def _iter_as_bytecode_as_instructions_py2(co):
    code = co.co_code
    op_offset_to_line = dict(dis.findlinestarts(co))
    labels = set(dis.findlabels(code))
    bytecode_len = len(code)
    i = 0
    extended_arg = 0
    free = None
    op_to_name = opname
    while i < bytecode_len:
        c = code[i]
        op = ord(c)
        is_jump_target = i in labels
        curr_op_name = op_to_name[op]
        initial_bytecode_offset = i
        i = i + 1
        if op < HAVE_ARGUMENT:
            yield _Instruction(curr_op_name, op, _get_line(op_offset_to_line, initial_bytecode_offset, 0), None, is_jump_target, initial_bytecode_offset, '')
        else:
            oparg = ord(code[i]) + ord(code[i + 1]) * 256 + extended_arg
            extended_arg = 0
            i = i + 2
            if op == EXTENDED_ARG:
                extended_arg = oparg * 65536
            if op in hasconst:
                yield _Instruction(curr_op_name, op, _get_line(op_offset_to_line, initial_bytecode_offset, 0), co.co_consts[oparg], is_jump_target, initial_bytecode_offset, repr(co.co_consts[oparg]))
            elif op in hasname:
                yield _Instruction(curr_op_name, op, _get_line(op_offset_to_line, initial_bytecode_offset, 0), co.co_names[oparg], is_jump_target, initial_bytecode_offset, str(co.co_names[oparg]))
            elif op in hasjrel:
                argval = i + oparg
                yield _Instruction(curr_op_name, op, _get_line(op_offset_to_line, initial_bytecode_offset, 0), argval, is_jump_target, initial_bytecode_offset, 'to ' + repr(argval))
            elif op in haslocal:
                yield _Instruction(curr_op_name, op, _get_line(op_offset_to_line, initial_bytecode_offset, 0), co.co_varnames[oparg], is_jump_target, initial_bytecode_offset, str(co.co_varnames[oparg]))
            elif op in hascompare:
                yield _Instruction(curr_op_name, op, _get_line(op_offset_to_line, initial_bytecode_offset, 0), cmp_op[oparg], is_jump_target, initial_bytecode_offset, cmp_op[oparg])
            elif op in hasfree:
                if free is None:
                    free = co.co_cellvars + co.co_freevars
                yield _Instruction(curr_op_name, op, _get_line(op_offset_to_line, initial_bytecode_offset, 0), free[oparg], is_jump_target, initial_bytecode_offset, str(free[oparg]))
            else:
                yield _Instruction(curr_op_name, op, _get_line(op_offset_to_line, initial_bytecode_offset, 0), oparg, is_jump_target, initial_bytecode_offset, str(oparg))