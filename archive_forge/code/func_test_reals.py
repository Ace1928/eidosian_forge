import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_reals(self):
    c = ir.Constant(flt, 1.5)
    self.assertEqual(str(c), 'float 0x3ff8000000000000')
    c = ir.Constant(flt, -1.5)
    self.assertEqual(str(c), 'float 0xbff8000000000000')
    c = ir.Constant(dbl, 1.5)
    self.assertEqual(str(c), 'double 0x3ff8000000000000')
    c = ir.Constant(dbl, -1.5)
    self.assertEqual(str(c), 'double 0xbff8000000000000')
    c = ir.Constant(dbl, ir.Undefined)
    self.assertEqual(str(c), 'double undef')
    c = ir.Constant(dbl, None)
    self.assertEqual(str(c), 'double 0.0')