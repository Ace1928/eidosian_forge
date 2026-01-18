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
def visit_IfExp(self, node):
    cond = self.visit(node.test)
    if _is_triton_tensor(cond):
        cond = cond.to(language.int1, _builder=self.builder)
        with enter_sub_region(self):
            ip, last_loc = self._get_insertion_point_and_loc()
            then_block = self.builder.create_block()
            self.builder.set_insertion_point_to_start(then_block)
            then_val = language.core._to_tensor(self.visit(node.body), self.builder)
            then_block = self.builder.get_insertion_block()
            else_block = self.builder.create_block()
            self.builder.set_insertion_point_to_start(else_block)
            else_val = language.core._to_tensor(self.visit(node.orelse), self.builder)
            else_block = self.builder.get_insertion_block()
            self._set_insertion_point_and_loc(ip, last_loc)
            assert then_val.type == else_val.type, f'ternary expression with dynamic condition has inconsistent types {then_val.type} and {else_val.type}'
            ret_type = then_val.type
            ret_type_ir = [ret_type.to_ir(self.builder)] if ret_type != language.void else []
            if_op = self.builder.create_if_op(ret_type_ir, cond.handle, True)
            then_block.merge_block_before(if_op.get_then_block())
            if ret_type_ir:
                self.builder.set_insertion_point_to_end(if_op.get_then_block())
                self.builder.create_yield_op([then_val.handle])
            self.builder.set_insertion_point_to_end(if_op.get_then_block())
            else_block.merge_block_before(if_op.get_else_block())
            if ret_type_ir:
                self.builder.set_insertion_point_to_end(if_op.get_else_block())
                self.builder.create_yield_op([else_val.handle])
            return language.core.tensor(if_op.get_result(0), ret_type) if ret_type_ir else None
    else:
        cond = _unwrap_if_constexpr(cond)
        if type(cond) not in _condition_types:
            raise UnsupportedLanguageConstruct(None, node, '`if` conditionals can only accept values of type {{{}}}, not objects of type {}'.format(', '.join((_.__name__ for _ in _condition_types)), type(cond).__name__))
        if cond:
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)