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
def _get_fn_file_line(fn):
    base_fn = fn
    while not isinstance(base_fn, JITFunction):
        base_fn = base_fn.fn
    file_name = base_fn.fn.__code__.co_filename
    lines, begin_line = inspect.getsourcelines(base_fn.fn)
    for idx, line in enumerate(lines):
        if line.strip().startswith('def '):
            begin_line += idx
            break
    return (file_name, begin_line)