import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def _scan_namedblocks(self, bc, cfa):
    """Scan namedblocks as denoted by a LOAD_GLOBAL bytecode referring
        to global variables with the pattern "SET_BLOCK_<name>", where "<name>"
        would be the name for the current block.
        """
    namedblocks = {}
    blocks = sorted([x.offset for x in cfa.iterblocks()])
    prefix = 'SET_BLOCK_'
    for inst in bc:
        if inst.opname == 'LOAD_GLOBAL':
            gv = bc.co_names[_fix_LOAD_GLOBAL_arg(inst.arg)]
            if gv.startswith(prefix):
                name = gv[len(prefix):]
                for s, e in zip(blocks, blocks[1:] + [blocks[-1] + 1]):
                    if s <= inst.offset < e:
                        break
                else:
                    raise AssertionError('unreachable loop')
                blkno = s
                namedblocks[name] = blkno
    return namedblocks