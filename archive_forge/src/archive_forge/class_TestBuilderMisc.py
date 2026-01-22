import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
class TestBuilderMisc(TestBase):
    """
    Test various other features of the IRBuilder class.
    """

    def test_attributes(self):
        block = self.block(name='start')
        builder = ir.IRBuilder(block)
        self.assertIs(builder.function, block.parent)
        self.assertIsInstance(builder.function, ir.Function)
        self.assertIs(builder.module, block.parent.module)
        self.assertIsInstance(builder.module, ir.Module)

    def test_goto_block(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        builder.add(a, b, 'c')
        bb_new = builder.append_basic_block(name='foo')
        with builder.goto_block(bb_new):
            builder.fadd(a, b, 'd')
            with builder.goto_entry_block():
                builder.sub(a, b, 'e')
            builder.fsub(a, b, 'f')
            builder.branch(bb_new)
        builder.mul(a, b, 'g')
        with builder.goto_block(bb_new):
            builder.fmul(a, b, 'h')
        self.check_block(block, '            my_block:\n                %"c" = add i32 %".1", %".2"\n                %"e" = sub i32 %".1", %".2"\n                %"g" = mul i32 %".1", %".2"\n            ')
        self.check_block(bb_new, '            foo:\n                %"d" = fadd i32 %".1", %".2"\n                %"f" = fsub i32 %".1", %".2"\n                %"h" = fmul i32 %".1", %".2"\n                br label %"foo"\n            ')

    def test_if_then(self):
        block = self.block(name='one')
        builder = ir.IRBuilder(block)
        z = ir.Constant(int1, 0)
        a = builder.add(z, z, 'a')
        with builder.if_then(a) as bbend:
            builder.add(z, z, 'b')
        self.assertIs(builder.block, bbend)
        c = builder.add(z, z, 'c')
        with builder.if_then(c):
            builder.add(z, z, 'd')
            builder.branch(block)
        self.check_func_body(builder.function, '            one:\n                %"a" = add i1 0, 0\n                br i1 %"a", label %"one.if", label %"one.endif"\n            one.if:\n                %"b" = add i1 0, 0\n                br label %"one.endif"\n            one.endif:\n                %"c" = add i1 0, 0\n                br i1 %"c", label %"one.endif.if", label %"one.endif.endif"\n            one.endif.if:\n                %"d" = add i1 0, 0\n                br label %"one"\n            one.endif.endif:\n            ')

    def test_if_then_nested(self):
        block = self.block(name='one')
        builder = ir.IRBuilder(block)
        z = ir.Constant(int1, 0)
        a = builder.add(z, z, 'a')
        with builder.if_then(a):
            b = builder.add(z, z, 'b')
            with builder.if_then(b):
                builder.add(z, z, 'c')
        builder.ret_void()
        self.check_func_body(builder.function, '            one:\n                %"a" = add i1 0, 0\n                br i1 %"a", label %"one.if", label %"one.endif"\n            one.if:\n                %"b" = add i1 0, 0\n                br i1 %"b", label %"one.if.if", label %"one.if.endif"\n            one.endif:\n                ret void\n            one.if.if:\n                %"c" = add i1 0, 0\n                br label %"one.if.endif"\n            one.if.endif:\n                br label %"one.endif"\n            ')

    def test_if_then_long_label(self):
        full_label = 'Long' * 20
        block = self.block(name=full_label)
        builder = ir.IRBuilder(block)
        z = ir.Constant(int1, 0)
        a = builder.add(z, z, 'a')
        with builder.if_then(a):
            b = builder.add(z, z, 'b')
            with builder.if_then(b):
                builder.add(z, z, 'c')
        builder.ret_void()
        self.check_func_body(builder.function, '            {full_label}:\n                %"a" = add i1 0, 0\n                br i1 %"a", label %"{label}.if", label %"{label}.endif"\n            {label}.if:\n                %"b" = add i1 0, 0\n                br i1 %"b", label %"{label}.if.if", label %"{label}.if.endif"\n            {label}.endif:\n                ret void\n            {label}.if.if:\n                %"c" = add i1 0, 0\n                br label %"{label}.if.endif"\n            {label}.if.endif:\n                br label %"{label}.endif"\n            '.format(full_label=full_label, label=full_label[:25] + '..'))

    def test_if_then_likely(self):

        def check(likely):
            block = self.block(name='one')
            builder = ir.IRBuilder(block)
            z = ir.Constant(int1, 0)
            with builder.if_then(z, likely=likely):
                pass
            self.check_block(block, '                one:\n                    br i1 0, label %"one.if", label %"one.endif", !prof !0\n                ')
            return builder
        builder = check(True)
        self.check_metadata(builder.module, '            !0 = !{ !"branch_weights", i32 99, i32 1 }\n            ')
        builder = check(False)
        self.check_metadata(builder.module, '            !0 = !{ !"branch_weights", i32 1, i32 99 }\n            ')

    def test_if_else(self):
        block = self.block(name='one')
        builder = ir.IRBuilder(block)
        z = ir.Constant(int1, 0)
        a = builder.add(z, z, 'a')
        with builder.if_else(a) as (then, otherwise):
            with then:
                builder.add(z, z, 'b')
            with otherwise:
                builder.add(z, z, 'c')
        with builder.if_else(a) as (then, otherwise):
            with then:
                builder.branch(block)
            with otherwise:
                builder.ret_void()
        self.check_func_body(builder.function, '            one:\n                %"a" = add i1 0, 0\n                br i1 %"a", label %"one.if", label %"one.else"\n            one.if:\n                %"b" = add i1 0, 0\n                br label %"one.endif"\n            one.else:\n                %"c" = add i1 0, 0\n                br label %"one.endif"\n            one.endif:\n                br i1 %"a", label %"one.endif.if", label %"one.endif.else"\n            one.endif.if:\n                br label %"one"\n            one.endif.else:\n                ret void\n            one.endif.endif:\n            ')

    def test_if_else_likely(self):

        def check(likely):
            block = self.block(name='one')
            builder = ir.IRBuilder(block)
            z = ir.Constant(int1, 0)
            with builder.if_else(z, likely=likely) as (then, otherwise):
                with then:
                    builder.branch(block)
                with otherwise:
                    builder.ret_void()
            self.check_func_body(builder.function, '                one:\n                    br i1 0, label %"one.if", label %"one.else", !prof !0\n                one.if:\n                    br label %"one"\n                one.else:\n                    ret void\n                one.endif:\n                ')
            return builder
        builder = check(True)
        self.check_metadata(builder.module, '            !0 = !{ !"branch_weights", i32 99, i32 1 }\n            ')
        builder = check(False)
        self.check_metadata(builder.module, '            !0 = !{ !"branch_weights", i32 1, i32 99 }\n            ')

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

    def test_instruction_removal(self):
        func = self.function()
        builder = ir.IRBuilder()
        blk = func.append_basic_block(name='entry')
        builder.position_at_end(blk)
        k = ir.Constant(int32, 1234)
        a = builder.add(k, k, 'a')
        retvoid = builder.ret_void()
        self.assertTrue(blk.is_terminated)
        builder.remove(retvoid)
        self.assertFalse(blk.is_terminated)
        b = builder.mul(a, a, 'b')
        c = builder.add(b, b, 'c')
        builder.remove(c)
        builder.ret_void()
        self.assertTrue(blk.is_terminated)
        self.check_block(blk, '            entry:\n                %"a" = add i32 1234, 1234\n                %"b" = mul i32 %"a", %"a"\n                ret void\n        ')

    def test_metadata(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        builder.debug_metadata = builder.module.add_metadata([])
        builder.alloca(ir.PointerType(int32), name='c')
        self.check_block(block, '            my_block:\n                %"c" = alloca i32*, !dbg !0\n            ')