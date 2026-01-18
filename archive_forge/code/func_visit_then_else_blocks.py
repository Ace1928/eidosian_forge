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
def visit_then_else_blocks(self, node, liveins, then_block, else_block):
    self.builder.set_insertion_point_to_start(then_block)
    self.visit_compound_statement(node.body)
    then_block = self.builder.get_insertion_block()
    then_defs = self.local_defs.copy()
    else_defs = {}
    if node.orelse:
        self.builder.set_insertion_point_to_start(else_block)
        self.lscope = liveins.copy()
        self.local_defs = {}
        self.visit_compound_statement(node.orelse)
        else_defs = self.local_defs.copy()
        else_block = self.builder.get_insertion_block()
    names = []
    ret_types = []
    ir_ret_types = []
    for name in liveins:
        for defs, block_name in [(then_defs, 'then'), (else_defs, 'else')]:
            if name in defs:
                assert defs[name].type == liveins[name].type, f'initial value for `{name}` is of type {liveins[name].type}, but the {block_name} block redefines it as {defs[name].type}'
        if name in then_defs or name in else_defs:
            names.append(name)
            ret_types.append(then_defs[name].type if name in then_defs else else_defs[name].type)
            ir_ret_types.append(then_defs[name].handle.get_type() if name in then_defs else else_defs[name].handle.get_type())
        if name in then_defs and name not in else_defs:
            else_defs[name] = liveins[name]
        if name in else_defs and name not in then_defs:
            then_defs[name] = liveins[name]
    for name in then_defs.keys() & else_defs.keys():
        if name in names:
            continue
        then_ty = then_defs[name].type
        else_ty = else_defs[name].type
        assert then_ty == else_ty, f'mismatched type for {name} between then block ({then_ty}) and else block ({else_ty})'
        names.append(name)
        ret_types.append(then_ty)
        ir_ret_types.append(then_defs[name].handle.get_type())
    return (then_defs, else_defs, then_block, else_block, names, ret_types, ir_ret_types)