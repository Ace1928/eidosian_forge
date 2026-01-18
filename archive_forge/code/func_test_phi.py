import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_phi(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a, b = builder.function.args[:2]
    bb2 = builder.function.append_basic_block('b2')
    bb3 = builder.function.append_basic_block('b3')
    phi = builder.phi(int32, 'my_phi', flags=('fast',))
    phi.add_incoming(a, bb2)
    phi.add_incoming(b, bb3)
    self.assertFalse(block.is_terminated)
    self.check_block(block, '            my_block:\n                %"my_phi" = phi fast i32 [%".1", %"b2"], [%".2", %"b3"]\n            ')