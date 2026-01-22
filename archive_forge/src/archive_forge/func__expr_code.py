from __future__ import annotations
import re
from typing import (
def _expr_code(self, expr: str) -> str:
    """Generate a Python expression for `expr`."""
    if '|' in expr:
        pipes = expr.split('|')
        code = self._expr_code(pipes[0])
        for func in pipes[1:]:
            self._variable(func, self.all_vars)
            code = f'c_{func}({code})'
    elif '.' in expr:
        dots = expr.split('.')
        code = self._expr_code(dots[0])
        args = ', '.join((repr(d) for d in dots[1:]))
        code = f'do_dots({code}, {args})'
    else:
        self._variable(expr, self.all_vars)
        code = 'c_%s' % expr
    return code