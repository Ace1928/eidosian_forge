import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_vector_ops(self):
    block = self.block(name='insert_block')
    builder = ir.IRBuilder(block)
    a, b = builder.function.args[:2]
    a.name = 'a'
    b.name = 'b'
    vecty = ir.VectorType(a.type, 2)
    vec = ir.Constant(vecty, ir.Undefined)
    idxty = ir.IntType(32)
    vec = builder.insert_element(vec, a, idxty(0), name='vec1')
    vec = builder.insert_element(vec, b, idxty(1), name='vec2')
    self.check_block(block, 'insert_block:\n    %"vec1" = insertelement <2 x i32> <i32 undef, i32 undef>, i32 %"a", i32 0\n    %"vec2" = insertelement <2 x i32> %"vec1", i32 %"b", i32 1\n            ')
    block = builder.append_basic_block('shuffle_block')
    builder.branch(block)
    builder.position_at_end(block)
    mask = ir.Constant(vecty, [1, 0])
    builder.shuffle_vector(vec, vec, mask, name='shuf')
    self.check_block(block, '            shuffle_block:\n                %"shuf" = shufflevector <2 x i32> %"vec2", <2 x i32> %"vec2", <2 x i32> <i32 1, i32 0>\n            ')
    block = builder.append_basic_block('add_block')
    builder.branch(block)
    builder.position_at_end(block)
    builder.add(vec, vec, name='sum')
    self.check_block(block, '            add_block:\n                %"sum" = add <2 x i32> %"vec2", %"vec2"\n            ')
    block = builder.append_basic_block('extract_block')
    builder.branch(block)
    builder.position_at_end(block)
    c = builder.extract_element(vec, idxty(0), name='ex1')
    d = builder.extract_element(vec, idxty(1), name='ex2')
    self.check_block(block, '            extract_block:\n              %"ex1" = extractelement <2 x i32> %"vec2", i32 0\n              %"ex2" = extractelement <2 x i32> %"vec2", i32 1\n            ')
    builder.ret(builder.add(c, d))
    self.assert_valid_ir(builder.module)