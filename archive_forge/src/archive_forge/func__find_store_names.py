import sys
import types
import collections
import io
from opcode import *
from opcode import (
def _find_store_names(co):
    """Find names of variables which are written in the code

    Generate sequence of strings
    """
    STORE_OPS = {opmap['STORE_NAME'], opmap['STORE_GLOBAL']}
    names = co.co_names
    for _, op, arg in _unpack_opargs(co.co_code):
        if op in STORE_OPS:
            yield names[arg]