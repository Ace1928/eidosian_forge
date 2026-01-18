from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
Printing helper function for ``Pow``

        Notes
        =====

        This preprocesses the ``sqrt`` as math formatter and prints division

        Examples
        ========

        >>> from sympy import sqrt
        >>> from sympy.printing.pycode import PythonCodePrinter
        >>> from sympy.abc import x

        Python code printer automatically looks up ``math.sqrt``.

        >>> printer = PythonCodePrinter()
        >>> printer._hprint_Pow(sqrt(x), rational=True)
        'x**(1/2)'
        >>> printer._hprint_Pow(sqrt(x), rational=False)
        'math.sqrt(x)'
        >>> printer._hprint_Pow(1/sqrt(x), rational=True)
        'x**(-1/2)'
        >>> printer._hprint_Pow(1/sqrt(x), rational=False)
        '1/math.sqrt(x)'
        >>> printer._hprint_Pow(1/x, rational=False)
        '1/x'
        >>> printer._hprint_Pow(1/x, rational=True)
        'x**(-1)'

        Using sqrt from numpy or mpmath

        >>> printer._hprint_Pow(sqrt(x), sqrt='numpy.sqrt')
        'numpy.sqrt(x)'
        >>> printer._hprint_Pow(sqrt(x), sqrt='mpmath.sqrt')
        'mpmath.sqrt(x)'

        See Also
        ========

        sympy.printing.str.StrPrinter._print_Pow
        