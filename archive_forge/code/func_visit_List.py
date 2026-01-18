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
def visit_List(self, node):
    ctx = self.visit(node.ctx)
    assert ctx is None
    elts = [self.visit(elt) for elt in node.elts]
    return elts