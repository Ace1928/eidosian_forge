import ast
import inspect
import re
import sys
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from .. import language
from .._C.libtriton.triton import ir
from ..language import constexpr, tensor
from ..runtime import JITFunction
from .errors import (CompilationError, CompileTimeAssertionFailure, UnsupportedLanguageConstruct)
def visit_For(self, node):
    IteratorClass = self.visit(node.iter.func)
    iter_args = [self.visit(arg) for arg in node.iter.args]
    if IteratorClass == language.static_range:
        iterator = IteratorClass(*iter_args)
        static_range = range(iterator.start.value, iterator.end.value, iterator.step.value)
        for i in static_range:
            self.lscope[node.target.id] = constexpr(i)
            self.visit_compound_statement(node.body)
            for stmt in node.orelse:
                ast.NodeVisitor.generic_visit(self, stmt)
        return
    if IteratorClass is not range:
        raise RuntimeError('Only `range` and `static_range` iterators are currently supported')
    lb = iter_args[0] if len(iter_args) > 1 else self.visit(ast.Num(0))
    ub = iter_args[1] if len(iter_args) > 1 else self.visit(node.iter.args[0])
    step = iter_args[2] if len(iter_args) > 2 else self.visit(ast.Num(1))
    negative_step = False
    if _is_constexpr(step) and step.value < 0:
        step = constexpr(-step.value)
        negative_step = True
        lb, ub = (ub, lb)
    lb = language.core._to_tensor(lb, self.builder)
    ub = language.core._to_tensor(ub, self.builder)
    step = language.core._to_tensor(step, self.builder)
    if not lb.dtype.is_int() or not ub.dtype.is_int() or (not step.dtype.is_int()):
        raise TypeError(f'For loop bounds and step must all be ints, are ({lb.dtype}, {ub.dtype}, {step.dtype})')
    iv_type = language.semantic.integer_promote_impl(lb.dtype, ub.dtype)
    iv_type = language.semantic.integer_promote_impl(iv_type, step.dtype)
    iv_ir_type = iv_type.to_ir(self.builder)
    iv_is_signed = iv_type.int_signedness == language.core.dtype.SIGNEDNESS.SIGNED
    lb = lb.handle
    ub = ub.handle
    step = step.handle
    lb = self.builder.create_int_cast(lb, iv_ir_type, iv_is_signed)
    ub = self.builder.create_int_cast(ub, iv_ir_type, iv_is_signed)
    step = self.builder.create_int_cast(step, iv_ir_type, iv_is_signed)
    iv = self.builder.create_undef(iv_ir_type)
    self.set_value(node.target.id, language.core.tensor(iv, iv_type))
    with enter_sub_region(self) as sr:
        liveins, insert_block = sr
        ip, last_loc = self._get_insertion_point_and_loc()
        block = self.builder.create_block()
        self.builder.set_insertion_point_to_start(block)
        self.scf_stack.append(node)
        self.visit_compound_statement(node.body)
        self.scf_stack.pop()
        block.erase()
        init_args = []
        yields = []
        names = []
        for name in self.local_defs:
            if name in liveins:
                assert _is_triton_tensor(self.local_defs[name]), f'{name} is not tensor'
                assert _is_triton_tensor(liveins[name])
                assert self.local_defs[name].type == liveins[name].type, f'Loop-carried variable {name} has initial type {liveins[name].type} but is re-assigned to {self.local_defs[name].type} in loop! Please make sure that the type stays consistent.'
                names.append(name)
                init_args.append(language.core._to_tensor(liveins[name], self.builder))
                yields.append(language.core._to_tensor(self.local_defs[name], self.builder))
        self._set_insertion_point_and_loc(ip, last_loc)
        for_op = self.builder.create_for_op(lb, ub, step, [arg.handle for arg in init_args])
        self.scf_stack.append(node)
        self.builder.set_insertion_point_to_start(for_op.get_body(0))
        for i, name in enumerate(names):
            self.set_value(name, language.core.tensor(for_op.get_body(0).arg(i + 1), yields[i].type))
        self.visit_compound_statement(node.body)
        self.scf_stack.pop()
        yields = []
        for name in self.local_defs:
            if name in liveins:
                yields.append(language.core._to_tensor(self.local_defs[name], self.builder))
        if len(yields) > 0:
            self.builder.create_yield_op([y.handle for y in yields])
        for_op_region = for_op.get_body(0).get_parent()
        assert for_op_region.size() == 1, 'We use SCF, so the loop body should only have one block'
        self.builder.set_insertion_point_to_start(for_op.get_body(0))
        iv = for_op.get_induction_var()
        if negative_step:
            iv = self.builder.create_sub(ub, iv)
            iv = self.builder.create_add(iv, lb)
        self.lscope[node.target.id].handle.replace_all_uses_with(iv)
        self.set_value(node.target.id, language.core.tensor(iv, iv_type))
    for i, name in enumerate(names):
        self.set_value(name, language.core.tensor(for_op.get_result(i), yields[i].type))
    for stmt in node.orelse:
        assert False, "Don't know what to do with else after for"
        ast.NodeVisitor.generic_visit(self, stmt)