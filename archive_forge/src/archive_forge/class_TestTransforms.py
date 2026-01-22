import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
class TestTransforms(TestBase):

    def test_call_transform(self):
        mod = ir.Module()
        foo = ir.Function(mod, ir.FunctionType(ir.VoidType(), ()), 'foo')
        bar = ir.Function(mod, ir.FunctionType(ir.VoidType(), ()), 'bar')
        builder = ir.IRBuilder()
        builder.position_at_end(foo.append_basic_block())
        call = builder.call(foo, ())
        self.assertEqual(call.callee, foo)
        modified = ir.replace_all_calls(mod, foo, bar)
        self.assertIn(call, modified)
        self.assertNotEqual(call.callee, foo)
        self.assertEqual(call.callee, bar)