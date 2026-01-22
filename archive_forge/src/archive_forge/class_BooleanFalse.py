from collections import defaultdict
from itertools import chain, combinations, product, permutations
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.function import Application, Derivative
from sympy.core.kind import BooleanKind, NumberKind
from sympy.core.numbers import Number
from sympy.core.operations import LatticeOp
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.utilities.iterables import sift, ibin
from sympy.utilities.misc import filldedent
class BooleanFalse(BooleanAtom, metaclass=Singleton):
    """
    SymPy version of ``False``, a singleton that can be accessed via ``S.false``.

    This is the SymPy version of ``False``, for use in the logic module. The
    primary advantage of using ``false`` instead of ``False`` is that shorthand
    Boolean operations like ``~`` and ``>>`` will work as expected on this class,
    whereas with ``False`` they act bitwise on 0. Functions in the logic module
    will return this class when they evaluate to false.

    Notes
    ======

    See the notes section in :py:class:`sympy.logic.boolalg.BooleanTrue`

    Examples
    ========

    >>> from sympy import sympify, true, false, Or
    >>> sympify(False)
    False
    >>> _ is False, _ is false
    (False, True)

    >>> Or(true, false)
    True
    >>> _ is true
    True

    Python operators give a boolean result for false but a
    bitwise result for False

    >>> ~false, ~False
    (True, -1)
    >>> false >> false, False >> False
    (True, 0)

    See Also
    ========

    sympy.logic.boolalg.BooleanTrue

    """

    def __bool__(self):
        return False

    def __hash__(self):
        return hash(False)

    def __eq__(self, other):
        if other is True:
            return False
        if other is False:
            return True
        return super().__eq__(other)

    @property
    def negated(self):
        return true

    def as_set(self):
        """
        Rewrite logic operators and relationals in terms of real sets.

        Examples
        ========

        >>> from sympy import false
        >>> false.as_set()
        EmptySet
        """
        return S.EmptySet