import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
class TestBlock(TestBase):

    def test_attributes(self):
        func = self.function()
        block = ir.Block(parent=func, name='start')
        self.assertIs(block.parent, func)
        self.assertFalse(block.is_terminated)

    def test_descr(self):
        block = self.block(name='my_block')
        self.assertEqual(self.descr(block), 'my_block:\n')
        block.instructions.extend(['a', 'b'])
        self.assertEqual(self.descr(block), 'my_block:\n  a\n  b\n')

    def test_replace(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        c = builder.add(a, b, 'c')
        d = builder.sub(a, b, 'd')
        builder.mul(d, b, 'e')
        f = ir.Instruction(block, a.type, 'sdiv', (c, b), 'f')
        self.check_block(block, '            my_block:\n                %"c" = add i32 %".1", %".2"\n                %"d" = sub i32 %".1", %".2"\n                %"e" = mul i32 %"d", %".2"\n            ')
        block.replace(d, f)
        self.check_block(block, '            my_block:\n                %"c" = add i32 %".1", %".2"\n                %"f" = sdiv i32 %"c", %".2"\n                %"e" = mul i32 %"f", %".2"\n            ')

    def test_repr(self):
        """
        Blocks should have a useful repr()
        """
        func = self.function()
        block = ir.Block(parent=func, name='start')
        self.assertEqual(repr(block), "<ir.Block 'start' of type 'label'>")