import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_positioning(self):
    """
        Test IRBuilder.position_{before,after,at_start,at_end}.
        """
    func = self.function()
    builder = ir.IRBuilder()
    z = ir.Constant(int32, 0)
    bb_one = func.append_basic_block(name='one')
    bb_two = func.append_basic_block(name='two')
    bb_three = func.append_basic_block(name='three')
    builder.position_at_start(bb_one)
    builder.add(z, z, 'a')
    builder.position_at_end(bb_two)
    builder.add(z, z, 'm')
    builder.add(z, z, 'n')
    builder.position_at_start(bb_two)
    o = builder.add(z, z, 'o')
    builder.add(z, z, 'p')
    builder.position_at_end(bb_one)
    b = builder.add(z, z, 'b')
    builder.position_after(o)
    builder.add(z, z, 'q')
    builder.position_before(b)
    builder.add(z, z, 'c')
    self.check_block(bb_one, '            one:\n                %"a" = add i32 0, 0\n                %"c" = add i32 0, 0\n                %"b" = add i32 0, 0\n            ')
    self.check_block(bb_two, '            two:\n                %"o" = add i32 0, 0\n                %"q" = add i32 0, 0\n                %"p" = add i32 0, 0\n                %"m" = add i32 0, 0\n                %"n" = add i32 0, 0\n            ')
    self.check_block(bb_three, '            three:\n            ')