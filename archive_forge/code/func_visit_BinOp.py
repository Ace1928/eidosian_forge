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
def visit_BinOp(self, node):
    lhs = self.visit(node.left)
    rhs = self.visit(node.right)
    method_name = self._method_name_for_bin_op.get(type(node.op))
    if method_name is None:
        raise UnsupportedLanguageConstruct(None, node, "AST binary operator '{}' is not (currently) implemented.".format(node.op.__name__))
    return self._apply_binary_method(method_name, lhs, rhs)