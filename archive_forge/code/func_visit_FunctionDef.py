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
def visit_FunctionDef(self, node):
    arg_names, kwarg_names = self.visit(node.args)
    if self.fn:
        raise UnsupportedLanguageConstruct(None, node, 'nested function definition is not supported.')
    for i, default_value in enumerate(node.args.defaults):
        arg_node = node.args.args[-i - 1]
        annotation = arg_node.annotation
        name = arg_node.arg
        st_target = ast.Name(id=name, ctx=ast.Store())
        if annotation is None:
            init_node = ast.Assign(targets=[st_target], value=default_value)
        else:
            init_node = ast.AnnAssign(target=st_target, value=default_value, annotation=annotation)
        self.visit(init_node)
    visibility = 'public' if self.is_kernel else 'private'
    self.fn = self.builder.get_or_insert_function(self.module, self.function_name, self.prototype.to_ir(self.builder), visibility, self.noinline)
    self.module.push_back(self.fn)
    entry = self.fn.add_entry_block()
    arg_values = []
    idx = 0
    for i, arg_name in enumerate(arg_names):
        if i in self.constants:
            cst = self.constants[i]
            if not _is_constexpr(cst):
                cst = constexpr(self.constants[i])
            arg_values.append(cst)
            continue
        else:
            if i in self.attributes:
                for name, value in self.attributes[i]:
                    self.fn.set_arg_attr(idx, name, value)
            arg_values.append(tensor(self.fn.args(idx), self.prototype.param_types[idx]))
            idx += 1
    insert_pt = self.builder.get_insertion_block()
    for arg_name, arg_value in zip(arg_names, arg_values):
        self.set_value(arg_name, arg_value)
    self.builder.set_insertion_point_to_start(entry)
    self.visit_compound_statement(node.body)
    if self.last_ret_type is None:
        self.builder.ret([])
    elif isinstance(self.last_ret_type, tuple):
        self.prototype.ret_types = list(self.last_ret_type)
        self.fn.reset_type(self.prototype.to_ir(self.builder))
    else:
        self.prototype.ret_types = [self.last_ret_type]
        self.fn.reset_type(self.prototype.to_ir(self.builder))
    if insert_pt:
        self.builder.set_insertion_point_to_end(insert_pt)
    self.fn.finalize()