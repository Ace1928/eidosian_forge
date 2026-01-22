from __future__ import annotations
from typing import Callable
from functools import reduce
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.numbers import igcdex, Integer
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core.cache import cacheit
Lazily evaluated table for $\cos \frac{\pi}{n}$ in square roots for
    $n \in \{3, 5, 17, 257, 65537\}$.

    Notes
    =====

    65537 is the only other known Fermat prime and it is nearly impossible to
    build in the current SymPy due to performance issues.

    References
    ==========

    https://r-knott.surrey.ac.uk/Fibonacci/simpleTrig.html
    