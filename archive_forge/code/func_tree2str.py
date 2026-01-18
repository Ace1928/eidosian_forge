import re
from sympy.core.numbers import (I, NumberSymbol, oo, zoo)
from sympy.core.symbol import Symbol
from sympy.utilities.iterables import numbered_symbols
from sympy.external import import_module
import warnings
@classmethod
def tree2str(cls, tree):
    """Converts a tree to string without translations.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> from sympy import sin
        >>> from sympy.plotting.experimental_lambdify import Lambdifier
        >>> str2tree = Lambdifier([x], x).str2tree
        >>> tree2str = Lambdifier([x], x).tree2str

        >>> tree2str(str2tree(str(x+y*sin(z)+1)))
        'x + y*sin(z) + 1'
        """
    if isinstance(tree, str):
        return tree
    else:
        return ''.join(map(cls.tree2str, tree))