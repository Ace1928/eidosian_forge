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
def visit_Compare(self, node):
    if not (len(node.comparators) == 1 and len(node.ops) == 1):
        raise UnsupportedLanguageConstruct(None, node, 'simultaneous multiple comparison is not supported')
    lhs = self.visit(node.left)
    rhs = self.visit(node.comparators[0])
    lhs_value = _unwrap_if_constexpr(lhs)
    rhs_value = _unwrap_if_constexpr(rhs)
    if type(node.ops[0]) == ast.Is:
        return constexpr(lhs_value is rhs_value)
    if type(node.ops[0]) == ast.IsNot:
        return constexpr(lhs_value is not rhs_value)
    method_name = self._method_name_for_comp_op.get(type(node.ops[0]))
    if method_name is None:
        raise UnsupportedLanguageConstruct(None, node, "AST comparison operator '{}' is not (currently) implemented.".format(node.ops[0].__name__))
    return self._apply_binary_method(method_name, lhs, rhs)