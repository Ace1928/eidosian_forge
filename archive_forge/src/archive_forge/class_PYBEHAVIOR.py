from __future__ import annotations
import os
import platform
import sys
from typing import Any, Iterable
class PYBEHAVIOR:
    """Flags indicating this Python's behavior."""
    pep626 = PYVERSION > (3, 10, 0, 'alpha', 4)
    optimize_if_debug = not pep626
    if pep626:
        optimize_if_not_debug = 1
    elif PYPY:
        if PYVERSION >= (3, 9):
            optimize_if_not_debug = 2
        else:
            optimize_if_not_debug = 3
    else:
        optimize_if_not_debug = 2
    docstring_only_function = not PYPY and PYVERSION <= (3, 10)
    finally_jumps_back = PYVERSION < (3, 10)
    trace_decorator_line_again = CPYTHON and PYVERSION > (3, 11, 0, 'alpha', 3, 0)
    report_absolute_files = (CPYTHON or (PYPY and PYPYVERSION >= (7, 3, 10))) and PYVERSION >= (3, 9)
    omit_after_jump = pep626 or (PYPY and PYVERSION >= (3, 9) and (PYPYVERSION >= (7, 3, 12)))
    omit_after_return = omit_after_jump or PYPY
    optimize_unreachable_try_else = pep626
    module_firstline_1 = pep626
    keep_constant_test = pep626
    exit_through_with = PYVERSION >= (3, 10, 0, 'beta')
    match_case = PYVERSION >= (3, 10)
    soft_keywords = PYVERSION >= (3, 10)
    empty_is_empty = PYVERSION >= (3, 11, 0, 'beta', 4)
    comprehensions_are_functions = PYVERSION <= (3, 12, 0, 'alpha', 7, 0)
    pep669 = bool(getattr(sys, 'monitoring', None))
    lasti_is_yield = PYVERSION < (3, 13)