import sys
import types
import collections
import io
from opcode import *
from opcode import (
def _find_imports(co):
    """Find import statements in the code

    Generate triplets (name, level, fromlist) where
    name is the imported module and level, fromlist are
    the corresponding args to __import__.
    """
    IMPORT_NAME = opmap['IMPORT_NAME']
    LOAD_CONST = opmap['LOAD_CONST']
    consts = co.co_consts
    names = co.co_names
    opargs = [(op, arg) for _, op, arg in _unpack_opargs(co.co_code) if op != EXTENDED_ARG]
    for i, (op, oparg) in enumerate(opargs):
        if op == IMPORT_NAME and i >= 2:
            from_op = opargs[i - 1]
            level_op = opargs[i - 2]
            if from_op[0] in hasconst and level_op[0] in hasconst:
                level = _get_const_value(level_op[0], level_op[1], consts)
                fromlist = _get_const_value(from_op[0], from_op[1], consts)
                yield (names[oparg], level, fromlist)