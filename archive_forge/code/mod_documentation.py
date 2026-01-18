from .add import Add
from .exprtools import gcd_terms
from .function import Function
from .kind import NumberKind
from .logic import fuzzy_and, fuzzy_not
from .mul import Mul
from .numbers import equal_valued
from .singleton import S
Try to return p % q if both are numbers or +/-p is known
            to be less than or equal q.
            