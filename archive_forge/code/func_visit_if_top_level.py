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
def visit_if_top_level(self, cond, node):
    has_endif_block = True
    with enter_sub_region(self) as sr:
        liveins, ip_block = sr
        then_block = self.builder.create_block()
        else_block = self.builder.create_block()
        endif_block = self.builder.create_block()
        self.builder.set_insertion_point_to_end(ip_block)
        self.builder.create_cond_branch(cond.handle, then_block, else_block)
        then_defs, else_defs, then_block, else_block, names, ret_types, ir_ret_types = self.visit_then_else_blocks(node, liveins, then_block, else_block)
        self.builder.set_insertion_point_to_end(then_block)
        if then_block.has_return() and else_block.has_return():
            has_endif_block = False
            endif_block.erase()
        if not then_block.has_terminator() and has_endif_block:
            self.builder.create_branch(endif_block, [then_defs[n].handle for n in names])
        self.builder.set_insertion_point_to_end(else_block)
        if not else_block.has_terminator() and has_endif_block:
            self.builder.create_branch(endif_block, [else_defs[n].handle for n in names])
        if has_endif_block:
            for ty in ir_ret_types:
                endif_block.add_argument(ty)
    if has_endif_block:
        self.builder.set_insertion_point_to_start(endif_block)
        for i, name in enumerate(names):
            new_tensor = language.core.tensor(endif_block.arg(i), ret_types[i])
            self.set_value(name, new_tensor)