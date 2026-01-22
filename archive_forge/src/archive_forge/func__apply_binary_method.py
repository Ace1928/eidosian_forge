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
def _apply_binary_method(self, method_name, lhs, rhs):
    if _is_triton_tensor(lhs):
        return getattr(lhs, method_name)(rhs, _builder=self.builder)
    if _is_triton_tensor(rhs):
        reverse_method_name = re.sub('__(.*)__', '__r\\1__', method_name)
        return getattr(rhs, reverse_method_name)(lhs, _builder=self.builder)
    return getattr(lhs, method_name)(rhs)