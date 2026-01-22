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
def _visit_function(self, fn) -> bool:
    if isinstance(fn, JITFunction) and (not fn.noinline):
        fn_node = fn.parse()
        return ContainsReturnChecker(self.gscope).visit(fn_node)
    return False