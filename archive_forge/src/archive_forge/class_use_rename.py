from sympy.codegen.ast import (
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Str
from sympy.core.sympify import sympify
from sympy.logic import true, false
from sympy.utilities.iterables import iterable
class use_rename(Token):
    """ Represents a renaming in a use statement in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import use_rename, use
    >>> from sympy import fcode
    >>> ren = use_rename("thingy", "convolution2d")
    >>> print(fcode(ren, source_format='free'))
    thingy => convolution2d
    >>> full = use('signallib', only=['snr', ren])
    >>> print(fcode(full, source_format='free'))
    use signallib, only: snr, thingy => convolution2d

    """
    __slots__ = _fields = ('local', 'original')
    _construct_local = String
    _construct_original = String