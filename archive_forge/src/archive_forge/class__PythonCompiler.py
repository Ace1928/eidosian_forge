from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
class _PythonCompiler(_Compiler):
    """Compiles an expression to Python."""
    compile_and = _binary_compiler('(%s and %s)')
    compile_or = _binary_compiler('(%s or %s)')
    compile_not = _unary_compiler('(not %s)')
    compile_mod = _binary_compiler('MOD(%s, %s)')

    def compile_relation(self, method, expr, range_list):
        ranges = ','.join([f'({self.compile(a)}, {self.compile(b)})' for a, b in range_list[1]])
        return f'{method.upper()}({self.compile(expr)}, [{ranges}])'