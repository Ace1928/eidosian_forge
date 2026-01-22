import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
class TestBuildInstructions(TestBase):
    """
    Test IR generation of LLVM instructions through the IRBuilder class.
    """
    maxDiff = 4000

    def test_simple(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        inst = builder.add(a, b, 'res')
        self.check_block(block, '            my_block:\n                %"res" = add i32 %".1", %".2"\n            ')
        self.assertEqual(repr(inst), "<ir.Instruction 'res' of type 'i32', opname 'add', operands (<ir.Argument '.1' of type i32>, <ir.Argument '.2' of type i32>)>")

    def test_binops(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b, ff = builder.function.args[:3]
        builder.add(a, b, 'c')
        builder.fadd(a, b, 'd')
        builder.sub(a, b, 'e')
        builder.fsub(a, b, 'f')
        builder.mul(a, b, 'g')
        builder.fmul(a, b, 'h')
        builder.udiv(a, b, 'i')
        builder.sdiv(a, b, 'j')
        builder.fdiv(a, b, 'k')
        builder.urem(a, b, 'l')
        builder.srem(a, b, 'm')
        builder.frem(a, b, 'n')
        builder.or_(a, b, 'o')
        builder.and_(a, b, 'p')
        builder.xor(a, b, 'q')
        builder.shl(a, b, 'r')
        builder.ashr(a, b, 's')
        builder.lshr(a, b, 't')
        with self.assertRaises(ValueError) as cm:
            builder.add(a, ff)
        self.assertEqual(str(cm.exception), 'Operands must be the same type, got (i32, double)')
        self.assertFalse(block.is_terminated)
        self.check_block(block, '            my_block:\n                %"c" = add i32 %".1", %".2"\n                %"d" = fadd i32 %".1", %".2"\n                %"e" = sub i32 %".1", %".2"\n                %"f" = fsub i32 %".1", %".2"\n                %"g" = mul i32 %".1", %".2"\n                %"h" = fmul i32 %".1", %".2"\n                %"i" = udiv i32 %".1", %".2"\n                %"j" = sdiv i32 %".1", %".2"\n                %"k" = fdiv i32 %".1", %".2"\n                %"l" = urem i32 %".1", %".2"\n                %"m" = srem i32 %".1", %".2"\n                %"n" = frem i32 %".1", %".2"\n                %"o" = or i32 %".1", %".2"\n                %"p" = and i32 %".1", %".2"\n                %"q" = xor i32 %".1", %".2"\n                %"r" = shl i32 %".1", %".2"\n                %"s" = ashr i32 %".1", %".2"\n                %"t" = lshr i32 %".1", %".2"\n            ')

    def test_binop_flags(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        builder.add(a, b, 'c', flags=('nuw',))
        builder.sub(a, b, 'd', flags=['nuw', 'nsw'])
        self.check_block(block, '            my_block:\n                %"c" = add nuw i32 %".1", %".2"\n                %"d" = sub nuw nsw i32 %".1", %".2"\n            ')

    def test_binop_fastmath_flags(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        builder.fadd(a, b, 'c', flags=('fast',))
        builder.fsub(a, b, 'd', flags=['ninf', 'nsz'])
        self.check_block(block, '            my_block:\n                %"c" = fadd fast i32 %".1", %".2"\n                %"d" = fsub ninf nsz i32 %".1", %".2"\n            ')

    def test_binops_with_overflow(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        builder.sadd_with_overflow(a, b, 'c')
        builder.smul_with_overflow(a, b, 'd')
        builder.ssub_with_overflow(a, b, 'e')
        builder.uadd_with_overflow(a, b, 'f')
        builder.umul_with_overflow(a, b, 'g')
        builder.usub_with_overflow(a, b, 'h')
        self.check_block(block, 'my_block:\n    %"c" = call {i32, i1} @"llvm.sadd.with.overflow.i32"(i32 %".1", i32 %".2")\n    %"d" = call {i32, i1} @"llvm.smul.with.overflow.i32"(i32 %".1", i32 %".2")\n    %"e" = call {i32, i1} @"llvm.ssub.with.overflow.i32"(i32 %".1", i32 %".2")\n    %"f" = call {i32, i1} @"llvm.uadd.with.overflow.i32"(i32 %".1", i32 %".2")\n    %"g" = call {i32, i1} @"llvm.umul.with.overflow.i32"(i32 %".1", i32 %".2")\n    %"h" = call {i32, i1} @"llvm.usub.with.overflow.i32"(i32 %".1", i32 %".2")\n            ')

    def test_unary_ops(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b, c = builder.function.args[:3]
        builder.neg(a, 'd')
        builder.not_(b, 'e')
        builder.fneg(c, 'f')
        self.assertFalse(block.is_terminated)
        self.check_block(block, '            my_block:\n                %"d" = sub i32 0, %".1"\n                %"e" = xor i32 %".2", -1\n                %"f" = fneg double %".3"\n            ')

    def test_replace_operand(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        undef1 = ir.Constant(ir.IntType(32), ir.Undefined)
        undef2 = ir.Constant(ir.IntType(32), ir.Undefined)
        c = builder.add(undef1, undef2, 'c')
        self.check_block(block, '            my_block:\n                %"c" = add i32 undef, undef\n            ')
        c.replace_usage(undef1, a)
        c.replace_usage(undef2, b)
        self.check_block(block, '            my_block:\n                %"c" = add i32 %".1", %".2"\n            ')

    def test_integer_comparisons(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        builder.icmp_unsigned('==', a, b, 'c')
        builder.icmp_unsigned('!=', a, b, 'd')
        builder.icmp_unsigned('<', a, b, 'e')
        builder.icmp_unsigned('<=', a, b, 'f')
        builder.icmp_unsigned('>', a, b, 'g')
        builder.icmp_unsigned('>=', a, b, 'h')
        builder.icmp_signed('==', a, b, 'i')
        builder.icmp_signed('!=', a, b, 'j')
        builder.icmp_signed('<', a, b, 'k')
        builder.icmp_signed('<=', a, b, 'l')
        builder.icmp_signed('>', a, b, 'm')
        builder.icmp_signed('>=', a, b, 'n')
        with self.assertRaises(ValueError):
            builder.icmp_signed('uno', a, b, 'zz')
        with self.assertRaises(ValueError):
            builder.icmp_signed('foo', a, b, 'zz')
        self.assertFalse(block.is_terminated)
        self.check_block(block, '            my_block:\n                %"c" = icmp eq i32 %".1", %".2"\n                %"d" = icmp ne i32 %".1", %".2"\n                %"e" = icmp ult i32 %".1", %".2"\n                %"f" = icmp ule i32 %".1", %".2"\n                %"g" = icmp ugt i32 %".1", %".2"\n                %"h" = icmp uge i32 %".1", %".2"\n                %"i" = icmp eq i32 %".1", %".2"\n                %"j" = icmp ne i32 %".1", %".2"\n                %"k" = icmp slt i32 %".1", %".2"\n                %"l" = icmp sle i32 %".1", %".2"\n                %"m" = icmp sgt i32 %".1", %".2"\n                %"n" = icmp sge i32 %".1", %".2"\n            ')

    def test_float_comparisons(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        builder.fcmp_ordered('==', a, b, 'c')
        builder.fcmp_ordered('!=', a, b, 'd')
        builder.fcmp_ordered('<', a, b, 'e')
        builder.fcmp_ordered('<=', a, b, 'f')
        builder.fcmp_ordered('>', a, b, 'g')
        builder.fcmp_ordered('>=', a, b, 'h')
        builder.fcmp_unordered('==', a, b, 'i')
        builder.fcmp_unordered('!=', a, b, 'j')
        builder.fcmp_unordered('<', a, b, 'k')
        builder.fcmp_unordered('<=', a, b, 'l')
        builder.fcmp_unordered('>', a, b, 'm')
        builder.fcmp_unordered('>=', a, b, 'n')
        builder.fcmp_ordered('ord', a, b, 'u')
        builder.fcmp_ordered('uno', a, b, 'v')
        builder.fcmp_unordered('ord', a, b, 'w')
        builder.fcmp_unordered('uno', a, b, 'x')
        builder.fcmp_unordered('olt', a, b, 'y', flags=['nnan', 'ninf', 'nsz', 'arcp', 'fast'])
        self.assertFalse(block.is_terminated)
        self.check_block(block, '            my_block:\n                %"c" = fcmp oeq i32 %".1", %".2"\n                %"d" = fcmp one i32 %".1", %".2"\n                %"e" = fcmp olt i32 %".1", %".2"\n                %"f" = fcmp ole i32 %".1", %".2"\n                %"g" = fcmp ogt i32 %".1", %".2"\n                %"h" = fcmp oge i32 %".1", %".2"\n                %"i" = fcmp ueq i32 %".1", %".2"\n                %"j" = fcmp une i32 %".1", %".2"\n                %"k" = fcmp ult i32 %".1", %".2"\n                %"l" = fcmp ule i32 %".1", %".2"\n                %"m" = fcmp ugt i32 %".1", %".2"\n                %"n" = fcmp uge i32 %".1", %".2"\n                %"u" = fcmp ord i32 %".1", %".2"\n                %"v" = fcmp uno i32 %".1", %".2"\n                %"w" = fcmp ord i32 %".1", %".2"\n                %"x" = fcmp uno i32 %".1", %".2"\n                %"y" = fcmp nnan ninf nsz arcp fast olt i32 %".1", %".2"\n            ')

    def test_misc_ops(self):
        block = self.block(name='my_block')
        t = ir.Constant(int1, True)
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        builder.select(t, a, b, 'c', flags=('arcp', 'nnan'))
        self.assertFalse(block.is_terminated)
        builder.unreachable()
        self.assertTrue(block.is_terminated)
        self.check_block(block, '            my_block:\n                %"c" = select arcp nnan i1 true, i32 %".1", i32 %".2"\n                unreachable\n            ')

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

    def test_mem_ops(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b, z = builder.function.args[:3]
        c = builder.alloca(int32, name='c')
        d = builder.alloca(int32, size=42, name='d')
        e = builder.alloca(dbl, size=a, name='e')
        e.align = 8
        self.assertEqual(e.type, ir.PointerType(dbl))
        ee = builder.store(z, e)
        self.assertEqual(ee.type, ir.VoidType())
        f = builder.store(b, c)
        self.assertEqual(f.type, ir.VoidType())
        g = builder.load(c, 'g')
        self.assertEqual(g.type, int32)
        h = builder.store(b, c, align=1)
        self.assertEqual(h.type, ir.VoidType())
        i = builder.load(c, 'i', align=1)
        self.assertEqual(i.type, int32)
        j = builder.store_atomic(b, c, ordering='seq_cst', align=4)
        self.assertEqual(j.type, ir.VoidType())
        k = builder.load_atomic(c, ordering='seq_cst', align=4, name='k')
        self.assertEqual(k.type, int32)
        with self.assertRaises(TypeError):
            builder.store(b, a)
        with self.assertRaises(TypeError):
            builder.load(b)
        with self.assertRaises(TypeError) as cm:
            builder.store(b, e)
        self.assertEqual(str(cm.exception), 'cannot store i32 to double*: mismatching types')
        self.check_block(block, '            my_block:\n                %"c" = alloca i32\n                %"d" = alloca i32, i32 42\n                %"e" = alloca double, i32 %".1", align 8\n                store double %".3", double* %"e"\n                store i32 %".2", i32* %"c"\n                %"g" = load i32, i32* %"c"\n                store i32 %".2", i32* %"c", align 1\n                %"i" = load i32, i32* %"c", align 1\n                store atomic i32 %".2", i32* %"c" seq_cst, align 4\n                %"k" = load atomic i32, i32* %"c" seq_cst, align 4\n            ')

    def test_gep(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        c = builder.alloca(ir.PointerType(int32), name='c')
        d = builder.gep(c, [ir.Constant(int32, 5), a], name='d')
        self.assertEqual(d.type, ir.PointerType(int32))
        self.check_block(block, '            my_block:\n                %"c" = alloca i32*\n                %"d" = getelementptr i32*, i32** %"c", i32 5, i32 %".1"\n            ')

    def test_gep_castinstr(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        int8ptr = int8.as_pointer()
        ls = ir.LiteralStructType([int64, int8ptr, int8ptr, int8ptr, int64])
        d = builder.bitcast(a, ls.as_pointer(), name='d')
        e = builder.gep(d, [ir.Constant(int32, x) for x in [0, 3]], name='e')
        self.assertEqual(e.type, ir.PointerType(int8ptr))
        self.check_block(block, '            my_block:\n                %"d" = bitcast i32 %".1" to {i64, i8*, i8*, i8*, i64}*\n                %"e" = getelementptr {i64, i8*, i8*, i8*, i64}, {i64, i8*, i8*, i8*, i64}* %"d", i32 0, i32 3\n            ')

    def test_gep_castinstr_addrspace(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        addrspace = 4
        int8ptr = int8.as_pointer()
        ls = ir.LiteralStructType([int64, int8ptr, int8ptr, int8ptr, int64])
        d = builder.bitcast(a, ls.as_pointer(addrspace=addrspace), name='d')
        e = builder.gep(d, [ir.Constant(int32, x) for x in [0, 3]], name='e')
        self.assertEqual(e.type.addrspace, addrspace)
        self.assertEqual(e.type, ir.PointerType(int8ptr, addrspace=addrspace))
        self.check_block(block, '            my_block:\n                %"d" = bitcast i32 %".1" to {i64, i8*, i8*, i8*, i64} addrspace(4)*\n                %"e" = getelementptr {i64, i8*, i8*, i8*, i64}, {i64, i8*, i8*, i8*, i64} addrspace(4)* %"d", i32 0, i32 3\n            ')

    def test_gep_addrspace(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        addrspace = 4
        c = builder.alloca(ir.PointerType(int32, addrspace=addrspace), name='c')
        self.assertEqual(str(c.type), 'i32 addrspace(4)**')
        self.assertEqual(c.type.pointee.addrspace, addrspace)
        d = builder.gep(c, [ir.Constant(int32, 5), a], name='d')
        self.assertEqual(d.type.addrspace, addrspace)
        e = builder.gep(d, [ir.Constant(int32, 10)], name='e')
        self.assertEqual(e.type.addrspace, addrspace)
        self.check_block(block, '            my_block:\n                %"c" = alloca i32 addrspace(4)*\n                %"d" = getelementptr i32 addrspace(4)*, i32 addrspace(4)** %"c", i32 5, i32 %".1"\n                %"e" = getelementptr i32, i32 addrspace(4)* %"d", i32 10\n            ')

    def test_extract_insert_value(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        tp_inner = ir.LiteralStructType([int32, int1])
        tp_outer = ir.LiteralStructType([int8, tp_inner])
        c_inner = ir.Constant(tp_inner, (ir.Constant(int32, 4), ir.Constant(int1, True)))
        c = builder.extract_value(c_inner, 0, name='c')
        d = builder.insert_value(c_inner, a, 0, name='d')
        e = builder.insert_value(d, ir.Constant(int1, False), 1, name='e')
        self.assertEqual(d.type, tp_inner)
        self.assertEqual(e.type, tp_inner)
        p_outer = builder.alloca(tp_outer, name='ptr')
        j = builder.load(p_outer, name='j')
        k = builder.extract_value(j, 0, name='k')
        l = builder.extract_value(j, 1, name='l')
        m = builder.extract_value(j, (1, 0), name='m')
        n = builder.extract_value(j, (1, 1), name='n')
        o = builder.insert_value(j, l, 1, name='o')
        p = builder.insert_value(j, a, (1, 0), name='p')
        self.assertEqual(k.type, int8)
        self.assertEqual(l.type, tp_inner)
        self.assertEqual(m.type, int32)
        self.assertEqual(n.type, int1)
        self.assertEqual(o.type, tp_outer)
        self.assertEqual(p.type, tp_outer)
        with self.assertRaises(TypeError):
            builder.extract_value(p_outer, 0)
        with self.assertRaises(TypeError):
            builder.extract_value(c_inner, (0, 0))
        with self.assertRaises(TypeError):
            builder.extract_value(c_inner, 5)
        with self.assertRaises(TypeError):
            builder.insert_value(a, b, 0)
        with self.assertRaises(TypeError):
            builder.insert_value(c_inner, a, 1)
        self.check_block(block, '            my_block:\n                %"c" = extractvalue {i32, i1} {i32 4, i1 true}, 0\n                %"d" = insertvalue {i32, i1} {i32 4, i1 true}, i32 %".1", 0\n                %"e" = insertvalue {i32, i1} %"d", i1 false, 1\n                %"ptr" = alloca {i8, {i32, i1}}\n                %"j" = load {i8, {i32, i1}}, {i8, {i32, i1}}* %"ptr"\n                %"k" = extractvalue {i8, {i32, i1}} %"j", 0\n                %"l" = extractvalue {i8, {i32, i1}} %"j", 1\n                %"m" = extractvalue {i8, {i32, i1}} %"j", 1, 0\n                %"n" = extractvalue {i8, {i32, i1}} %"j", 1, 1\n                %"o" = insertvalue {i8, {i32, i1}} %"j", {i32, i1} %"l", 1\n                %"p" = insertvalue {i8, {i32, i1}} %"j", i32 %".1", 1, 0\n            ')

    def test_cast_ops(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b, fa, ptr = builder.function.args[:4]
        c = builder.trunc(a, int8, name='c')
        d = builder.zext(c, int32, name='d')
        e = builder.sext(c, int32, name='e')
        fb = builder.fptrunc(fa, flt, 'fb')
        fc = builder.fpext(fb, dbl, 'fc')
        g = builder.fptoui(fa, int32, 'g')
        h = builder.fptosi(fa, int8, 'h')
        fd = builder.uitofp(g, flt, 'fd')
        fe = builder.sitofp(h, dbl, 'fe')
        i = builder.ptrtoint(ptr, int32, 'i')
        j = builder.inttoptr(i, ir.PointerType(int8), 'j')
        k = builder.bitcast(a, flt, 'k')
        self.assertFalse(block.is_terminated)
        self.check_block(block, '            my_block:\n                %"c" = trunc i32 %".1" to i8\n                %"d" = zext i8 %"c" to i32\n                %"e" = sext i8 %"c" to i32\n                %"fb" = fptrunc double %".3" to float\n                %"fc" = fpext float %"fb" to double\n                %"g" = fptoui double %".3" to i32\n                %"h" = fptosi double %".3" to i8\n                %"fd" = uitofp i32 %"g" to float\n                %"fe" = sitofp i8 %"h" to double\n                %"i" = ptrtoint i32* %".4" to i32\n                %"j" = inttoptr i32 %"i" to i8*\n                %"k" = bitcast i32 %".1" to float\n            ')

    def test_atomicrmw(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        c = builder.alloca(int32, name='c')
        d = builder.atomic_rmw('add', c, a, 'monotonic', 'd')
        self.assertEqual(d.type, int32)
        self.check_block(block, '            my_block:\n                %"c" = alloca i32\n                %"d" = atomicrmw add i32* %"c", i32 %".1" monotonic\n            ')

    def test_branch(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        bb_target = builder.function.append_basic_block(name='target')
        builder.branch(bb_target)
        self.assertTrue(block.is_terminated)
        self.check_block(block, '            my_block:\n                br label %"target"\n            ')

    def test_cbranch(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        bb_true = builder.function.append_basic_block(name='b_true')
        bb_false = builder.function.append_basic_block(name='b_false')
        builder.cbranch(ir.Constant(int1, False), bb_true, bb_false)
        self.assertTrue(block.is_terminated)
        self.check_block(block, '            my_block:\n                br i1 false, label %"b_true", label %"b_false"\n            ')

    def test_cbranch_weights(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        bb_true = builder.function.append_basic_block(name='b_true')
        bb_false = builder.function.append_basic_block(name='b_false')
        br = builder.cbranch(ir.Constant(int1, False), bb_true, bb_false)
        br.set_weights([5, 42])
        self.assertTrue(block.is_terminated)
        self.check_block(block, '            my_block:\n                br i1 false, label %"b_true", label %"b_false", !prof !0\n            ')
        self.check_metadata(builder.module, '            !0 = !{ !"branch_weights", i32 5, i32 42 }\n            ')

    def test_branch_indirect(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        bb_1 = builder.function.append_basic_block(name='b_1')
        bb_2 = builder.function.append_basic_block(name='b_2')
        indirectbr = builder.branch_indirect(ir.BlockAddress(builder.function, bb_1))
        indirectbr.add_destination(bb_1)
        indirectbr.add_destination(bb_2)
        self.assertTrue(block.is_terminated)
        self.check_block(block, '            my_block:\n                indirectbr i8* blockaddress(@"my_func", %"b_1"), [label %"b_1", label %"b_2"]\n            ')

    def test_returns(self):

        def check(block, expected_ir):
            self.assertTrue(block.is_terminated)
            self.check_block(block, expected_ir)
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        builder.ret_void()
        check(block, '            my_block:\n                ret void\n            ')
        block = self.block(name='other_block')
        builder = ir.IRBuilder(block)
        builder.ret(int32(5))
        check(block, '            other_block:\n                ret i32 5\n            ')
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        inst = builder.ret_void()
        inst.set_metadata('dbg', block.module.add_metadata(()))
        check(block, '            my_block:\n                ret void, !dbg !0\n            ')
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        inst = builder.ret(int32(6))
        inst.set_metadata('dbg', block.module.add_metadata(()))
        check(block, '            my_block:\n                ret i32 6, !dbg !0\n            ')

    def test_switch(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        bb_onzero = builder.function.append_basic_block(name='onzero')
        bb_onone = builder.function.append_basic_block(name='onone')
        bb_ontwo = builder.function.append_basic_block(name='ontwo')
        bb_else = builder.function.append_basic_block(name='otherwise')
        sw = builder.switch(a, bb_else)
        sw.add_case(ir.Constant(int32, 0), bb_onzero)
        sw.add_case(ir.Constant(int32, 1), bb_onone)
        sw.add_case(2, bb_ontwo)
        self.assertTrue(block.is_terminated)
        self.check_block(block, '            my_block:\n                switch i32 %".1", label %"otherwise" [i32 0, label %"onzero" i32 1, label %"onone" i32 2, label %"ontwo"]\n            ')

    def test_call(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        tp_f = ir.FunctionType(flt, (int32, int32))
        tp_g = ir.FunctionType(dbl, (int32,), var_arg=True)
        tp_h = ir.FunctionType(hlf, (int32, int32))
        f = ir.Function(builder.function.module, tp_f, 'f')
        g = ir.Function(builder.function.module, tp_g, 'g')
        h = ir.Function(builder.function.module, tp_h, 'h')
        builder.call(f, (a, b), 'res_f')
        builder.call(g, (b, a), 'res_g')
        builder.call(h, (a, b), 'res_h')
        builder.call(f, (a, b), 'res_f_fast', cconv='fastcc')
        res_f_readonly = builder.call(f, (a, b), 'res_f_readonly')
        res_f_readonly.attributes.add('readonly')
        builder.call(f, (a, b), 'res_fast', fastmath='fast')
        builder.call(f, (a, b), 'res_nnan_ninf', fastmath=('nnan', 'ninf'))
        builder.call(f, (a, b), 'res_noinline', attrs='noinline')
        builder.call(f, (a, b), 'res_alwaysinline', attrs='alwaysinline')
        builder.call(f, (a, b), 'res_noinline_ro', attrs=('noinline', 'readonly'))
        builder.call(f, (a, b), 'res_convergent', attrs='convergent')
        self.check_block(block, '        my_block:\n            %"res_f" = call float @"f"(i32 %".1", i32 %".2")\n            %"res_g" = call double (i32, ...) @"g"(i32 %".2", i32 %".1")\n            %"res_h" = call half @"h"(i32 %".1", i32 %".2")\n            %"res_f_fast" = call fastcc float @"f"(i32 %".1", i32 %".2")\n            %"res_f_readonly" = call float @"f"(i32 %".1", i32 %".2") readonly\n            %"res_fast" = call fast float @"f"(i32 %".1", i32 %".2")\n            %"res_nnan_ninf" = call ninf nnan float @"f"(i32 %".1", i32 %".2")\n            %"res_noinline" = call float @"f"(i32 %".1", i32 %".2") noinline\n            %"res_alwaysinline" = call float @"f"(i32 %".1", i32 %".2") alwaysinline\n            %"res_noinline_ro" = call float @"f"(i32 %".1", i32 %".2") noinline readonly\n            %"res_convergent" = call float @"f"(i32 %".1", i32 %".2") convergent\n        ')

    def test_call_metadata(self):
        """
        Function calls with metadata arguments.
        """
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        dbg_declare_ty = ir.FunctionType(ir.VoidType(), [ir.MetaDataType()] * 3)
        dbg_declare = ir.Function(builder.module, dbg_declare_ty, 'llvm.dbg.declare')
        a = builder.alloca(int32, name='a')
        b = builder.module.add_metadata(())
        builder.call(dbg_declare, (a, b, b))
        self.check_block(block, '            my_block:\n                %"a" = alloca i32\n                call void @"llvm.dbg.declare"(metadata i32* %"a", metadata !0, metadata !0)\n            ')

    def test_call_attributes(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        fun_ty = ir.FunctionType(ir.VoidType(), (int32.as_pointer(), int32, int32.as_pointer()))
        fun = ir.Function(builder.function.module, fun_ty, 'fun')
        fun.args[0].add_attribute('sret')
        retval = builder.alloca(int32, name='retval')
        other = builder.alloca(int32, name='other')
        builder.call(fun, (retval, ir.Constant(int32, 42), other), arg_attrs={0: ('sret', 'noalias'), 2: 'noalias'})
        self.check_block_regex(block, '        my_block:\n            %"retval" = alloca i32\n            %"other" = alloca i32\n            call void @"fun"\\(i32\\* noalias sret(\\(i32\\))? %"retval", i32 42, i32\\* noalias %"other"\\)\n        ')

    def test_call_tail(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        fun_ty = ir.FunctionType(ir.VoidType(), ())
        fun = ir.Function(builder.function.module, fun_ty, 'my_fun')
        builder.call(fun, ())
        builder.call(fun, (), tail=False)
        builder.call(fun, (), tail=True)
        builder.call(fun, (), tail='tail')
        builder.call(fun, (), tail='notail')
        builder.call(fun, (), tail='musttail')
        builder.call(fun, (), tail=[])
        builder.call(fun, (), tail='not a marker')
        self.check_block(block, '        my_block:\n            call void @"my_fun"()\n            call void @"my_fun"()\n            tail call void @"my_fun"()\n            tail call void @"my_fun"()\n            notail call void @"my_fun"()\n            musttail call void @"my_fun"()\n            call void @"my_fun"()\n            tail call void @"my_fun"()\n        ')

    def test_invalid_call_attributes(self):
        block = self.block()
        builder = ir.IRBuilder(block)
        fun_ty = ir.FunctionType(ir.VoidType(), ())
        fun = ir.Function(builder.function.module, fun_ty, 'fun')
        with self.assertRaises(ValueError):
            builder.call(fun, (), arg_attrs={0: 'sret'})

    def test_invoke(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        tp_f = ir.FunctionType(flt, (int32, int32))
        f = ir.Function(builder.function.module, tp_f, 'f')
        bb_normal = builder.function.append_basic_block(name='normal')
        bb_unwind = builder.function.append_basic_block(name='unwind')
        builder.invoke(f, (a, b), bb_normal, bb_unwind, 'res_f')
        self.check_block(block, '            my_block:\n                %"res_f" = invoke float @"f"(i32 %".1", i32 %".2")\n                    to label %"normal" unwind label %"unwind"\n            ')

    def test_invoke_attributes(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        fun_ty = ir.FunctionType(ir.VoidType(), (int32.as_pointer(), int32, int32.as_pointer()))
        fun = ir.Function(builder.function.module, fun_ty, 'fun')
        fun.calling_convention = 'fastcc'
        fun.args[0].add_attribute('sret')
        retval = builder.alloca(int32, name='retval')
        other = builder.alloca(int32, name='other')
        bb_normal = builder.function.append_basic_block(name='normal')
        bb_unwind = builder.function.append_basic_block(name='unwind')
        builder.invoke(fun, (retval, ir.Constant(int32, 42), other), bb_normal, bb_unwind, cconv='fastcc', fastmath='fast', attrs='noinline', arg_attrs={0: ('sret', 'noalias'), 2: 'noalias'})
        self.check_block_regex(block, '        my_block:\n            %"retval" = alloca i32\n            %"other" = alloca i32\n            invoke fast fastcc void @"fun"\\(i32\\* noalias sret(\\(i32\\))? %"retval", i32 42, i32\\* noalias %"other"\\) noinline\n                to label %"normal" unwind label %"unwind"\n        ')

    def test_landingpad(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        lp = builder.landingpad(ir.LiteralStructType([int32, int8.as_pointer()]), 'lp')
        int_typeinfo = ir.GlobalVariable(builder.function.module, int8.as_pointer(), '_ZTIi')
        int_typeinfo.global_constant = True
        lp.add_clause(ir.CatchClause(int_typeinfo))
        lp.add_clause(ir.FilterClause(ir.Constant(ir.ArrayType(int_typeinfo.type, 1), [int_typeinfo])))
        builder.resume(lp)
        self.check_block(block, '            my_block:\n                %"lp" = landingpad {i32, i8*}\n                    catch i8** @"_ZTIi"\n                    filter [1 x i8**] [i8** @"_ZTIi"]\n                resume {i32, i8*} %"lp"\n            ')

    def test_assume(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a, b = builder.function.args[:2]
        c = builder.icmp_signed('>', a, b, name='c')
        builder.assume(c)
        self.check_block(block, '            my_block:\n                %"c" = icmp sgt i32 %".1", %".2"\n                call void @"llvm.assume"(i1 %"c")\n            ')

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

    def test_bitreverse(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(int64, 5)
        c = builder.bitreverse(a, name='c')
        builder.ret(c)
        self.check_block(block, '            my_block:\n                %"c" = call i64 @"llvm.bitreverse.i64"(i64 5)\n                ret i64 %"c"\n            ')

    def test_bitreverse_wrongtype(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(flt, 5)
        with self.assertRaises(TypeError) as raises:
            builder.bitreverse(a, name='c')
        self.assertIn('expected an integer type, got float', str(raises.exception))

    def test_fence(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        with self.assertRaises(ValueError) as raises:
            builder.fence('monotonic', None)
        self.assertIn('Invalid fence ordering "monotonic"!', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            builder.fence(None, 'monotonic')
        self.assertIn('Invalid fence ordering "None"!', str(raises.exception))
        builder.fence('acquire', None)
        builder.fence('release', 'singlethread')
        builder.fence('acq_rel', 'singlethread')
        builder.fence('seq_cst')
        builder.ret_void()
        self.check_block(block, '            my_block:\n                fence acquire\n                fence syncscope("singlethread") release\n                fence syncscope("singlethread") acq_rel\n                fence seq_cst\n                ret void\n            ')

    def test_comment(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        with self.assertRaises(AssertionError):
            builder.comment('so\nmany lines')
        builder.comment('my comment')
        builder.ret_void()
        self.check_block(block, '            my_block:\n                ; my comment\n                ret void\n            ')

    def test_bswap(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(int32, 5)
        c = builder.bswap(a, name='c')
        builder.ret(c)
        self.check_block(block, '            my_block:\n                %"c" = call i32 @"llvm.bswap.i32"(i32 5)\n                ret i32 %"c"\n            ')

    def test_ctpop(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(int16, 5)
        c = builder.ctpop(a, name='c')
        builder.ret(c)
        self.check_block(block, '            my_block:\n                %"c" = call i16 @"llvm.ctpop.i16"(i16 5)\n                ret i16 %"c"\n            ')

    def test_ctlz(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(int16, 5)
        b = ir.Constant(int1, 1)
        c = builder.ctlz(a, b, name='c')
        builder.ret(c)
        self.check_block(block, '            my_block:\n                %"c" = call i16 @"llvm.ctlz.i16"(i16 5, i1 1)\n                ret i16 %"c"\n            ')

    def test_convert_to_fp16_f32(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(flt, 5.0)
        b = builder.convert_to_fp16(a, name='b')
        builder.ret(b)
        self.check_block(block, '            my_block:\n                %"b" = call i16 @"llvm.convert.to.fp16.f32"(float 0x4014000000000000)\n                ret i16 %"b"\n            ')

    def test_convert_to_fp16_f32_wrongtype(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(int16, 5)
        with self.assertRaises(TypeError) as raises:
            builder.convert_to_fp16(a, name='b')
        self.assertIn('expected a float type, got i16', str(raises.exception))

    def test_convert_from_fp16_f32(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(int16, 5)
        b = builder.convert_from_fp16(a, name='b', to=flt)
        builder.ret(b)
        self.check_block(block, '            my_block:\n                %"b" = call float @"llvm.convert.from.fp16.f32"(i16 5)\n                ret float %"b"\n            ')

    def test_convert_from_fp16_f32_notype(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(flt, 5.5)
        with self.assertRaises(TypeError) as raises:
            builder.convert_from_fp16(a, name='b')
        self.assertIn('expected a float return type', str(raises.exception))

    def test_convert_from_fp16_f32_wrongtype(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(flt, 5.5)
        with self.assertRaises(TypeError) as raises:
            builder.convert_from_fp16(a, name='b', to=flt)
        self.assertIn('expected an i16 type, got float', str(raises.exception))

    def test_convert_from_fp16_f32_wrongtype2(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(flt, 5.5)
        with self.assertRaises(TypeError) as raises:
            builder.convert_from_fp16(a, name='b', to=int16)
        self.assertIn('expected a float type, got i16', str(raises.exception))

    def test_cttz(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(int64, 5)
        b = ir.Constant(int1, 1)
        c = builder.cttz(a, b, name='c')
        builder.ret(c)
        self.check_block(block, '            my_block:\n                %"c" = call i64 @"llvm.cttz.i64"(i64 5, i1 1)\n                ret i64 %"c"\n            ')

    def test_cttz_wrongflag(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(int64, 5)
        b = ir.Constant(int32, 3)
        with self.assertRaises(TypeError) as raises:
            builder.cttz(a, b, name='c')
        self.assertIn('expected an i1 type, got i32', str(raises.exception))

    def test_cttz_wrongtype(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(flt, 5)
        b = ir.Constant(int1, 1)
        with self.assertRaises(TypeError) as raises:
            builder.cttz(a, b, name='c')
        self.assertIn('expected an integer type, got float', str(raises.exception))

    def test_fma(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(flt, 5)
        b = ir.Constant(flt, 1)
        c = ir.Constant(flt, 2)
        fma = builder.fma(a, b, c, name='fma')
        builder.ret(fma)
        self.check_block(block, '            my_block:\n                %"fma" = call float @"llvm.fma.f32"(float 0x4014000000000000, float 0x3ff0000000000000, float 0x4000000000000000)\n                ret float %"fma"\n            ')

    def test_fma_wrongtype(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(int32, 5)
        b = ir.Constant(int32, 1)
        c = ir.Constant(int32, 2)
        with self.assertRaises(TypeError) as raises:
            builder.fma(a, b, c, name='fma')
        self.assertIn('expected an floating point type, got i32', str(raises.exception))

    def test_fma_mixedtypes(self):
        block = self.block(name='my_block')
        builder = ir.IRBuilder(block)
        a = ir.Constant(flt, 5)
        b = ir.Constant(dbl, 1)
        c = ir.Constant(flt, 2)
        with self.assertRaises(TypeError) as raises:
            builder.fma(a, b, c, name='fma')
        self.assertIn('expected types to be the same, got float, double, float', str(raises.exception))

    def test_arg_attributes(self):

        def gen_code(attr_name):
            fnty = ir.FunctionType(ir.IntType(32), [ir.IntType(32).as_pointer(), ir.IntType(32)])
            module = ir.Module()
            func = ir.Function(module, fnty, name='sum')
            bb_entry = func.append_basic_block()
            bb_loop = func.append_basic_block()
            bb_exit = func.append_basic_block()
            builder = ir.IRBuilder()
            builder.position_at_end(bb_entry)
            builder.branch(bb_loop)
            builder.position_at_end(bb_loop)
            index = builder.phi(ir.IntType(32))
            index.add_incoming(ir.Constant(index.type, 0), bb_entry)
            accum = builder.phi(ir.IntType(32))
            accum.add_incoming(ir.Constant(accum.type, 0), bb_entry)
            func.args[0].add_attribute(attr_name)
            ptr = builder.gep(func.args[0], [index])
            value = builder.load(ptr)
            added = builder.add(accum, value)
            accum.add_incoming(added, bb_loop)
            indexp1 = builder.add(index, ir.Constant(index.type, 1))
            index.add_incoming(indexp1, bb_loop)
            cond = builder.icmp_unsigned('<', indexp1, func.args[1])
            builder.cbranch(cond, bb_loop, bb_exit)
            builder.position_at_end(bb_exit)
            builder.ret(added)
            return str(module)
        for attr_name in ('byref', 'byval', 'elementtype', 'immarg', 'inalloca', 'inreg', 'nest', 'noalias', 'nocapture', 'nofree', 'nonnull', 'noundef', 'preallocated', 'returned', 'signext', 'swiftasync', 'swifterror', 'swiftself', 'zeroext'):
            llvm.parse_assembly(gen_code(attr_name))