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
def visit_BoolOp(self, node: ast.BoolOp):
    if len(node.values) != 2:
        raise UnsupportedLanguageConstruct(None, node, 'chained boolean operators (A or B or C) are not supported; use parentheses to split the chain.')
    lhs = self.visit(node.values[0])
    rhs = self.visit(node.values[1])
    method_name = self._method_name_for_bool_op.get(type(node.op))
    if method_name is None:
        raise UnsupportedLanguageConstruct(None, node, "AST boolean operator '{}' is not (currently) implemented.".format(node.op.__name__))
    return self._apply_binary_method(method_name, lhs, rhs)