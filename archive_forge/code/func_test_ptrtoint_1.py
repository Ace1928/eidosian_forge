import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_ptrtoint_1(self):
    ptr = ir.Constant(int64.as_pointer(), None)
    one = ir.Constant(int32, 1)
    c = ptr.ptrtoint(int32)
    self.assertRaises(TypeError, one.ptrtoint, int64)
    self.assertRaises(TypeError, ptr.ptrtoint, flt)
    self.assertEqual(str(c), 'ptrtoint (i64* null to i32)')