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
def visit_JoinedStr(self, node):
    values = list(node.values)
    for i, value in enumerate(values):
        if isinstance(value, ast.Constant):
            values[i] = str(value.value)
        elif isinstance(value, ast.FormattedValue):
            conversion_code = value.conversion
            evaluated = self.visit(value.value)
            if not _is_constexpr(evaluated):
                raise UnsupportedLanguageConstruct(None, node, 'Cannot evaluate f-string containing non-constexpr conversion values, found conversion of type ' + str(type(evaluated)))
            values[i] = ('{}' if conversion_code < 0 else '{!' + chr(conversion_code) + '}').format(evaluated.value)
        else:
            raise AssertionError('encountered unexpected node of type {} in a JoinedStr node'.format(type(value)))
    return ''.join(values)