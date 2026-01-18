import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_sitofp(self):
    c = ir.Constant(int32, 1).sitofp(flt)
    self.assertEqual(str(c), 'sitofp (i32 1 to float)')