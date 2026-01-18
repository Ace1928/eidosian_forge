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
def visit(self, node):
    if node is None:
        return
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        warnings.simplefilter('ignore', PendingDeprecationWarning)
        self.last_node = node
        last_loc = self.builder.get_loc()
        if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
            self.builder.set_loc(self.file_name, self.begin_line + node.lineno, node.col_offset)
            last_loc = self.builder.get_loc()
        ret = super().visit(node)
        if last_loc:
            self.builder.set_loc(last_loc)
        return ret