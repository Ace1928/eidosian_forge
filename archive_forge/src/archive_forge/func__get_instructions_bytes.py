import sys
import types
import collections
import io
from opcode import *
from opcode import (
def _get_instructions_bytes(code, varname_from_oparg=None, names=None, co_consts=None, linestarts=None, line_offset=0, exception_entries=(), co_positions=None, show_caches=False):
    """Iterate over the instructions in a bytecode string.

    Generates a sequence of Instruction namedtuples giving the details of each
    opcode.  Additional information about the code's runtime environment
    (e.g. variable names, co_consts) can be specified using optional
    arguments.

    """
    co_positions = co_positions or iter(())
    get_name = None if names is None else names.__getitem__
    labels = set(findlabels(code))
    for start, end, target, _, _ in exception_entries:
        for i in range(start, end):
            labels.add(target)
    starts_line = None
    for offset, op, arg in _unpack_opargs(code):
        if linestarts is not None:
            starts_line = linestarts.get(offset, None)
            if starts_line is not None:
                starts_line += line_offset
        is_jump_target = offset in labels
        argval = None
        argrepr = ''
        positions = Positions(*next(co_positions, ()))
        deop = _deoptop(op)
        if arg is not None:
            argval = arg
            if deop in hasconst:
                argval, argrepr = _get_const_info(deop, arg, co_consts)
            elif deop in hasname:
                if deop == LOAD_GLOBAL:
                    argval, argrepr = _get_name_info(arg // 2, get_name)
                    if arg & 1 and argrepr:
                        argrepr = 'NULL + ' + argrepr
                else:
                    argval, argrepr = _get_name_info(arg, get_name)
            elif deop in hasjabs:
                argval = arg * 2
                argrepr = 'to ' + repr(argval)
            elif deop in hasjrel:
                signed_arg = -arg if _is_backward_jump(deop) else arg
                argval = offset + 2 + signed_arg * 2
                argrepr = 'to ' + repr(argval)
            elif deop in haslocal or deop in hasfree:
                argval, argrepr = _get_name_info(arg, varname_from_oparg)
            elif deop in hascompare:
                argval = cmp_op[arg]
                argrepr = argval
            elif deop == FORMAT_VALUE:
                argval, argrepr = FORMAT_VALUE_CONVERTERS[arg & 3]
                argval = (argval, bool(arg & 4))
                if argval[1]:
                    if argrepr:
                        argrepr += ', '
                    argrepr += 'with format'
            elif deop == MAKE_FUNCTION:
                argrepr = ', '.join((s for i, s in enumerate(MAKE_FUNCTION_FLAGS) if arg & 1 << i))
            elif deop == BINARY_OP:
                _, argrepr = _nb_ops[arg]
        yield Instruction(_all_opname[op], op, arg, argval, argrepr, offset, starts_line, is_jump_target, positions)
        caches = _inline_cache_entries[deop]
        if not caches:
            continue
        if not show_caches:
            for _ in range(caches):
                next(co_positions, ())
            continue
        for name, size in _cache_format[opname[deop]].items():
            for i in range(size):
                offset += 2
                if i == 0 and op != deop:
                    data = code[offset:offset + 2 * size]
                    argrepr = f'{name}: {int.from_bytes(data, sys.byteorder)}'
                else:
                    argrepr = ''
                yield Instruction('CACHE', CACHE, 0, None, argrepr, offset, None, False, Positions(*next(co_positions, ())))