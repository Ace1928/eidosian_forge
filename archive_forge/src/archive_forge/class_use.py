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
class use(Token):
    """ Represents a use statement in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import use
    >>> from sympy import fcode
    >>> fcode(use('signallib'), source_format='free')
    'use signallib'
    >>> fcode(use('signallib', [('metric', 'snr')]), source_format='free')
    'use signallib, metric => snr'
    >>> fcode(use('signallib', only=['snr', 'convolution2d']), source_format='free')
    'use signallib, only: snr, convolution2d'

    """
    __slots__ = _fields = ('namespace', 'rename', 'only')
    defaults = {'rename': none, 'only': none}
    _construct_namespace = staticmethod(_name)
    _construct_rename = staticmethod(lambda args: Tuple(*[arg if isinstance(arg, use_rename) else use_rename(*arg) for arg in args]))
    _construct_only = staticmethod(lambda args: Tuple(*[arg if isinstance(arg, use_rename) else _name(arg) for arg in args]))