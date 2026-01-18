import math
import sys
import unittest
from llvmlite.ir import (
from llvmlite.tests import TestCase
def test_struct_repr(self):
    tp = LiteralStructType([int8, int16])
    c = Constant(tp, (Constant(int8, 100), Constant(int16, 1000)))
    self.assertEqual(str(c), '{i8, i16} {i8 100, i16 1000}')