import enum
import dis
import opcode as _opcode
import sys
from marshal import dumps as _dumps
from _pydevd_frame_eval.vendored import bytecode as _bytecode
def _pushes_back(opname):
    if opname in ['CALL_FINALLY']:
        return False
    return (opname.startswith('UNARY_') or opname.startswith('GET_') or opname.startswith('BINARY_') or opname.startswith('INPLACE_') or opname.startswith('BUILD_') or opname.startswith('CALL_')) or opname in ('LIST_TO_TUPLE', 'LIST_EXTEND', 'SET_UPDATE', 'DICT_UPDATE', 'DICT_MERGE', 'IS_OP', 'CONTAINS_OP', 'FORMAT_VALUE', 'MAKE_FUNCTION', 'IMPORT_NAME', 'SET_ADD', 'LIST_APPEND', 'MAP_ADD', 'LOAD_ATTR')