from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import call_highest_priority
from sympy.core.parameters import global_parameters
from sympy.core.function import AppliedUndef, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S, Singleton
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.core.sympify import sympify
from sympy.matrices import Matrix
from sympy.polys import lcm, factor
from sympy.sets.sets import Interval, Intersection
from sympy.tensor.indexed import Idx
from sympy.utilities.iterables import flatten, is_sequence, iterable
class EmptySequence(SeqBase, metaclass=Singleton):
    """Represents an empty sequence.

    The empty sequence is also available as a singleton as
    ``S.EmptySequence``.

    Examples
    ========

    >>> from sympy import EmptySequence, SeqPer
    >>> from sympy.abc import x
    >>> EmptySequence
    EmptySequence
    >>> SeqPer((1, 2), (x, 0, 10)) + EmptySequence
    SeqPer((1, 2), (x, 0, 10))
    >>> SeqPer((1, 2)) * EmptySequence
    EmptySequence
    >>> EmptySequence.coeff_mul(-1)
    EmptySequence
    """

    @property
    def interval(self):
        return S.EmptySet

    @property
    def length(self):
        return S.Zero

    def coeff_mul(self, coeff):
        """See docstring of SeqBase.coeff_mul"""
        return self

    def __iter__(self):
        return iter([])