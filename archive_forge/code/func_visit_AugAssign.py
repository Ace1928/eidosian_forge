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
def visit_AugAssign(self, node):
    name = node.target.id
    lhs = ast.Name(id=name, ctx=ast.Load())
    rhs = ast.BinOp(lhs, node.op, node.value)
    assign = ast.Assign(targets=[node.target], value=rhs)
    self.visit(assign)
    return self.dereference_name(name)