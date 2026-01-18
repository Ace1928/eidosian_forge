from .z3 import *
from .z3core import *
from .z3printer import *
from fractions import Fraction
from .z3 import _get_ctx
 Return True if `self != other`.

        >>> Numeral(Sqrt(2)) != 2
        True
        >>> Numeral(Sqrt(3)) != Numeral(Sqrt(2))
        True
        >>> Numeral(Sqrt(2)) != Numeral(Sqrt(2))
        False
        