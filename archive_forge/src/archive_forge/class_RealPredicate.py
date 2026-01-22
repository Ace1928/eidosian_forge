from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class RealPredicate(Predicate):
    """
    Real number predicate.

    Explanation
    ===========

    ``Q.real(x)`` is true iff ``x`` is a real number, i.e., it is in the
    interval `(-\\infty, \\infty)`.  Note that, in particular the
    infinities are not real. Use ``Q.extended_real`` if you want to
    consider those as well.

    A few important facts about reals:

    - Every real number is positive, negative, or zero.  Furthermore,
        because these sets are pairwise disjoint, each real number is
        exactly one of those three.

    - Every real number is also complex.

    - Every real number is finite.

    - Every real number is either rational or irrational.

    - Every real number is either algebraic or transcendental.

    - The facts ``Q.negative``, ``Q.zero``, ``Q.positive``,
        ``Q.nonnegative``, ``Q.nonpositive``, ``Q.nonzero``,
        ``Q.integer``, ``Q.rational``, and ``Q.irrational`` all imply
        ``Q.real``, as do all facts that imply those facts.

    - The facts ``Q.algebraic``, and ``Q.transcendental`` do not imply
        ``Q.real``; they imply ``Q.complex``. An algebraic or
        transcendental number may or may not be real.

    - The "non" facts (i.e., ``Q.nonnegative``, ``Q.nonzero``,
        ``Q.nonpositive`` and ``Q.noninteger``) are not equivalent to
        not the fact, but rather, not the fact *and* ``Q.real``.
        For example, ``Q.nonnegative`` means ``~Q.negative & Q.real``.
        So for example, ``I`` is not nonnegative, nonzero, or
        nonpositive.

    Examples
    ========

    >>> from sympy import Q, ask, symbols
    >>> x = symbols('x')
    >>> ask(Q.real(x), Q.positive(x))
    True
    >>> ask(Q.real(0))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Real_number

    """
    name = 'real'
    handler = Dispatcher('RealHandler', doc='Handler for Q.real.\n\nTest that an expression belongs to the field of real numbers.')