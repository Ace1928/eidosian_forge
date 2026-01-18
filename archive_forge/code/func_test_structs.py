import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_structs(self):
    st1 = ir.LiteralStructType((flt, int1))
    st2 = ir.LiteralStructType((int32, st1))
    c = ir.Constant(st1, (ir.Constant(ir.FloatType(), 1.5), ir.Constant(int1, True)))
    self.assertEqual(str(c), '{float, i1} {float 0x3ff8000000000000, i1 true}')
    c = ir.Constant.literal_struct((ir.Constant(ir.FloatType(), 1.5), ir.Constant(int1, True)))
    self.assertEqual(c.type, st1)
    self.assertEqual(str(c), '{float, i1} {float 0x3ff8000000000000, i1 true}')
    c = ir.Constant.literal_struct((ir.Constant(ir.FloatType(), 1.5), ir.Constant(int1, ir.Undefined)))
    self.assertEqual(c.type, st1)
    self.assertEqual(str(c), '{float, i1} {float 0x3ff8000000000000, i1 undef}')
    c = ir.Constant(st1, ir.Undefined)
    self.assertEqual(str(c), '{float, i1} undef')
    c = ir.Constant(st1, None)
    self.assertEqual(str(c), '{float, i1} zeroinitializer')
    c1 = ir.Constant(st1, (1.5, True))
    self.assertEqual(str(c1), '{float, i1} {float 0x3ff8000000000000, i1 true}')
    c2 = ir.Constant(st2, (42, c1))
    self.assertEqual(str(c2), '{i32, {float, i1}} {i32 42, {float, i1} {float 0x3ff8000000000000, i1 true}}')
    c3 = ir.Constant(st2, (42, (1.5, True)))
    self.assertEqual(str(c3), str(c2))
    with self.assertRaises(ValueError):
        ir.Constant(st2, (4, 5, 6))