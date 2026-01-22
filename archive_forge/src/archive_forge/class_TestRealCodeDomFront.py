import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
class TestRealCodeDomFront(TestCase):
    """Test IDOM and DOMFRONT computation on real python bytecode.
    Note: there will be less testing on IDOM (esp in loop) because of
    the extra blocks inserted by the interpreter.  But, testing on DOMFRONT
    (which depends on IDOM) is easier.

    Testing is done by associating names to basicblock by using globals of
    the pattern "SET_BLOCK_<name>", which are scanned by
    `.get_cfa_and_namedblocks` into *namedblocks* dictionary.  That way, we
    can check that a block of a certain name is a IDOM or DOMFRONT of another
    named block.
    """

    def cfa(self, bc):
        cfa = ControlFlowAnalysis(bc)
        cfa.run()
        return cfa

    def get_cfa_and_namedblocks(self, fn):
        fid = FunctionIdentity.from_function(fn)
        bc = ByteCode(func_id=fid)
        cfa = self.cfa(bc)
        namedblocks = self._scan_namedblocks(bc, cfa)
        return (cfa, namedblocks)

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

    def test_loop(self):

        def foo(n):
            c = 0
            SET_BLOCK_A
            i = 0
            while SET_BLOCK_B0:
                SET_BLOCK_B1
                c += 1
                i += 1
            SET_BLOCK_C
            return c
        cfa, blkpts = self.get_cfa_and_namedblocks(foo)
        idoms = cfa.graph.immediate_dominators()
        if PYVERSION < (3, 10):
            self.assertEqual(blkpts['B0'], idoms[blkpts['B1']])
        domfront = cfa.graph.dominance_frontier()
        self.assertFalse(domfront[blkpts['A']])
        self.assertFalse(domfront[blkpts['C']])
        if PYVERSION < (3, 10):
            self.assertEqual({blkpts['B0']}, domfront[blkpts['B1']])

    def test_loop_nested_and_break(self):

        def foo(n):
            SET_BLOCK_A
            while SET_BLOCK_B0:
                SET_BLOCK_B1
                while SET_BLOCK_C0:
                    SET_BLOCK_C1
                    if SET_BLOCK_D0:
                        SET_BLOCK_D1
                        break
                    elif n:
                        SET_BLOCK_D2
                    SET_BLOCK_E
                SET_BLOCK_F
            SET_BLOCK_G
        cfa, blkpts = self.get_cfa_and_namedblocks(foo)
        idoms = cfa.graph.immediate_dominators()
        self.assertEqual(blkpts['D0'], blkpts['C1'])
        if PYVERSION < (3, 10):
            self.assertEqual(blkpts['C0'], idoms[blkpts['C1']])
        domfront = cfa.graph.dominance_frontier()
        self.assertFalse(domfront[blkpts['A']])
        self.assertFalse(domfront[blkpts['G']])
        if PYVERSION < (3, 10):
            self.assertEqual({blkpts['B0']}, domfront[blkpts['B1']])
        if PYVERSION < (3, 10):
            self.assertEqual({blkpts['C0'], blkpts['F']}, domfront[blkpts['C1']])
        self.assertEqual({blkpts['F']}, domfront[blkpts['D1']])
        self.assertEqual({blkpts['E']}, domfront[blkpts['D2']])
        if PYVERSION < (3, 10):
            self.assertEqual({blkpts['C0']}, domfront[blkpts['E']])
            self.assertEqual({blkpts['B0']}, domfront[blkpts['F']])
            self.assertEqual({blkpts['B0']}, domfront[blkpts['B0']])

    def test_if_else(self):

        def foo(a, b):
            c = 0
            SET_BLOCK_A
            if a < b:
                SET_BLOCK_B
                c = 1
            elif SET_BLOCK_C0:
                SET_BLOCK_C1
                c = 2
            else:
                SET_BLOCK_D
                c = 3
            SET_BLOCK_E
            if a % b == 0:
                SET_BLOCK_F
                c += 1
            SET_BLOCK_G
            return c
        cfa, blkpts = self.get_cfa_and_namedblocks(foo)
        idoms = cfa.graph.immediate_dominators()
        self.assertEqual(blkpts['A'], idoms[blkpts['B']])
        self.assertEqual(blkpts['A'], idoms[blkpts['C0']])
        self.assertEqual(blkpts['C0'], idoms[blkpts['C1']])
        self.assertEqual(blkpts['C0'], idoms[blkpts['D']])
        self.assertEqual(blkpts['A'], idoms[blkpts['E']])
        self.assertEqual(blkpts['E'], idoms[blkpts['F']])
        self.assertEqual(blkpts['E'], idoms[blkpts['G']])
        domfront = cfa.graph.dominance_frontier()
        self.assertFalse(domfront[blkpts['A']])
        self.assertFalse(domfront[blkpts['E']])
        self.assertFalse(domfront[blkpts['G']])
        self.assertEqual({blkpts['E']}, domfront[blkpts['B']])
        self.assertEqual({blkpts['E']}, domfront[blkpts['C0']])
        self.assertEqual({blkpts['E']}, domfront[blkpts['C1']])
        self.assertEqual({blkpts['E']}, domfront[blkpts['D']])
        self.assertEqual({blkpts['G']}, domfront[blkpts['F']])

    def test_if_else_nested(self):

        def foo():
            if SET_BLOCK_A0:
                SET_BLOCK_A1
                if SET_BLOCK_B0:
                    SET_BLOCK_B1
                    a = 0
                else:
                    if SET_BLOCK_C0:
                        SET_BLOCK_C1
                        a = 1
                    else:
                        SET_BLOCK_C2
                        a = 2
                    SET_BLOCK_D
                SET_BLOCK_E
            SET_BLOCK_F
            return a
        cfa, blkpts = self.get_cfa_and_namedblocks(foo)
        idoms = cfa.graph.immediate_dominators()
        self.assertEqual(blkpts['A0'], idoms[blkpts['A1']])
        self.assertEqual(blkpts['A1'], idoms[blkpts['B1']])
        self.assertEqual(blkpts['A1'], idoms[blkpts['C0']])
        self.assertEqual(blkpts['C0'], idoms[blkpts['D']])
        self.assertEqual(blkpts['A1'], idoms[blkpts['E']])
        self.assertEqual(blkpts['A0'], idoms[blkpts['F']])
        domfront = cfa.graph.dominance_frontier()
        self.assertFalse(domfront[blkpts['A0']])
        self.assertFalse(domfront[blkpts['F']])
        self.assertEqual({blkpts['E']}, domfront[blkpts['B1']])
        self.assertEqual({blkpts['D']}, domfront[blkpts['C1']])
        self.assertEqual({blkpts['E']}, domfront[blkpts['D']])
        self.assertEqual({blkpts['F']}, domfront[blkpts['E']])

    def test_infinite_loop(self):

        def foo():
            SET_BLOCK_A
            while True:
                if SET_BLOCK_B:
                    SET_BLOCK_C
                    return
                SET_BLOCK_D
            SET_BLOCK_E
        cfa, blkpts = self.get_cfa_and_namedblocks(foo)
        idoms = cfa.graph.immediate_dominators()
        if PYVERSION >= (3, 10):
            self.assertNotIn('E', blkpts)
        else:
            self.assertNotIn(blkpts['E'], idoms)
        self.assertEqual(blkpts['B'], idoms[blkpts['C']])
        self.assertEqual(blkpts['B'], idoms[blkpts['D']])
        domfront = cfa.graph.dominance_frontier()
        if PYVERSION < (3, 10):
            self.assertNotIn(blkpts['E'], domfront)
        self.assertFalse(domfront[blkpts['A']])
        self.assertFalse(domfront[blkpts['C']])
        self.assertEqual({blkpts['B']}, domfront[blkpts['B']])
        self.assertEqual({blkpts['B']}, domfront[blkpts['D']])