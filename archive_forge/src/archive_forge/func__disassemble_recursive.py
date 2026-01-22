import sys
import types
import collections
import io
from opcode import *
from opcode import (
def _disassemble_recursive(co, *, file=None, depth=None, show_caches=False, adaptive=False):
    disassemble(co, file=file, show_caches=show_caches, adaptive=adaptive)
    if depth is None or depth > 0:
        if depth is not None:
            depth = depth - 1
        for x in co.co_consts:
            if hasattr(x, 'co_code'):
                print(file=file)
                print('Disassembly of %r:' % (x,), file=file)
                _disassemble_recursive(x, file=file, depth=depth, show_caches=show_caches, adaptive=adaptive)