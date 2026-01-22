from __future__ import annotations
from sympy.core import Basic, S
from sympy.core.function import Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence
from functools import reduce
def _print_Function_with_args(self, func, func_args):
    if func in self.known_functions:
        cond_func = self.known_functions[func]
        func = None
        if isinstance(cond_func, str):
            func = cond_func
        else:
            for cond, func in cond_func:
                if cond(func_args):
                    break
        if func is not None:
            try:
                return func(*[self.parenthesize(item, 0) for item in func_args])
            except TypeError:
                return '{}({})'.format(func, self.stringify(func_args, ', '))
    elif isinstance(func, Lambda):
        return self._print(func(*func_args))
    else:
        return self._print_not_supported(func)