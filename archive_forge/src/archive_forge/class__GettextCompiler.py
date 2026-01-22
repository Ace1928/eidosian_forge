from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
class _GettextCompiler(_Compiler):
    """Compile into a gettext plural expression."""
    compile_i = _Compiler.compile_n
    compile_v = compile_zero
    compile_w = compile_zero
    compile_f = compile_zero
    compile_t = compile_zero

    def compile_relation(self, method, expr, range_list):
        rv = []
        expr = self.compile(expr)
        for item in range_list[1]:
            if item[0] == item[1]:
                rv.append(f'({expr} == {self.compile(item[0])})')
            else:
                min, max = map(self.compile, item)
                rv.append(f'({expr} >= {min} && {expr} <= {max})')
        return f'({' || '.join(rv)})'