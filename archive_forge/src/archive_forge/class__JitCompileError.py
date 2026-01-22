import ast
import collections
import inspect
import linecache
import numbers
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
import warnings
import types
import numpy
from cupy_backends.cuda.api import runtime
from cupy._core._codeblock import CodeBlock, _CodeType
from cupy._core import _kernel
from cupy._core._dtype import _raise_if_invalid_cast
from cupyx import jit
from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types
from cupyx.jit._internal_types import Data
from cupyx.jit._internal_types import Constant
from cupyx.jit import _builtin_funcs
from cupyx.jit import _interface
class _JitCompileError(Exception):

    def __init__(self, e, node):
        self.error_type = type(e)
        self.mes = str(e)
        self.node = node

    def reraise(self, pycode):
        start = self.node.lineno
        end = getattr(self.node, 'end_lineno', start)
        pycode = '\n'.join([(f'> {line}' if start <= i + 1 <= end else f'  {line}').rstrip() for i, line in enumerate(pycode.split('\n'))])
        raise self.error_type(self.mes + '\n\n' + pycode)