from .libmp.backend import xrange
from .libmp import int_types, sqrt_fixed

    Given a real number `x`, ``identify(x)`` attempts to find an exact
    formula for `x`. This formula is returned as a string. If no match
    is found, ``None`` is returned. With ``full=True``, a list of
    matching formulas is returned.

    As a simple example, :func:`~mpmath.identify` will find an algebraic
    formula for the golden ratio::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> identify(phi)
        '((1+sqrt(5))/2)'

    :func:`~mpmath.identify` can identify simple algebraic numbers and simple
    combinations of given base constants, as well as certain basic
    transformations thereof. More specifically, :func:`~mpmath.identify`
    looks for the following:

        1. Fractions
        2. Quadratic algebraic numbers
        3. Rational linear combinations of the base constants
        4. Any of the above after first transforming `x` into `f(x)` where
           `f(x)` is `1/x`, `\sqrt x`, `x^2`, `\log x` or `\exp x`, either
           directly or with `x` or `f(x)` multiplied or divided by one of
           the base constants
        5. Products of fractional powers of the base constants and
           small integers

    Base constants can be given as a list of strings representing mpmath
    expressions (:func:`~mpmath.identify` will ``eval`` the strings to numerical
    values and use the original strings for the output), or as a dict of
    formula:value pairs.

    In order not to produce spurious results, :func:`~mpmath.identify` should
    be used with high precision; preferably 50 digits or more.

    **Examples**

    Simple identifications can be performed safely at standard
    precision. Here the default recognition of rational, algebraic,
    and exp/log of algebraic numbers is demonstrated::

        >>> mp.dps = 15
        >>> identify(0.22222222222222222)
        '(2/9)'
        >>> identify(1.9662210973805663)
        'sqrt(((24+sqrt(48))/8))'
        >>> identify(4.1132503787829275)
        'exp((sqrt(8)/2))'
        >>> identify(0.881373587019543)
        'log(((2+sqrt(8))/2))'

    By default, :func:`~mpmath.identify` does not recognize `\pi`. At standard
    precision it finds a not too useful approximation. At slightly
    increased precision, this approximation is no longer accurate
    enough and :func:`~mpmath.identify` more correctly returns ``None``::

        >>> identify(pi)
        '(2**(176/117)*3**(20/117)*5**(35/39))/(7**(92/117))'
        >>> mp.dps = 30
        >>> identify(pi)
        >>>

    Numbers such as `\pi`, and simple combinations of user-defined
    constants, can be identified if they are provided explicitly::

        >>> identify(3*pi-2*e, ['pi', 'e'])
        '(3*pi + (-2)*e)'

    Here is an example using a dict of constants. Note that the
    constants need not be "atomic"; :func:`~mpmath.identify` can just
    as well express the given number in terms of expressions
    given by formulas::

        >>> identify(pi+e, {'a':pi+2, 'b':2*e})
        '((-2) + 1*a + (1/2)*b)'

    Next, we attempt some identifications with a set of base constants.
    It is necessary to increase the precision a bit.

        >>> mp.dps = 50
        >>> base = ['sqrt(2)','pi','log(2)']
        >>> identify(0.25, base)
        '(1/4)'
        >>> identify(3*pi + 2*sqrt(2) + 5*log(2)/7, base)
        '(2*sqrt(2) + 3*pi + (5/7)*log(2))'
        >>> identify(exp(pi+2), base)
        'exp((2 + 1*pi))'
        >>> identify(1/(3+sqrt(2)), base)
        '((3/7) + (-1/7)*sqrt(2))'
        >>> identify(sqrt(2)/(3*pi+4), base)
        'sqrt(2)/(4 + 3*pi)'
        >>> identify(5**(mpf(1)/3)*pi*log(2)**2, base)
        '5**(1/3)*pi*log(2)**2'

    An example of an erroneous solution being found when too low
    precision is used::

        >>> mp.dps = 15
        >>> identify(1/(3*pi-4*e+sqrt(8)), ['pi', 'e', 'sqrt(2)'])
        '((11/25) + (-158/75)*pi + (76/75)*e + (44/15)*sqrt(2))'
        >>> mp.dps = 50
        >>> identify(1/(3*pi-4*e+sqrt(8)), ['pi', 'e', 'sqrt(2)'])
        '1/(3*pi + (-4)*e + 2*sqrt(2))'

    **Finding approximate solutions**

    The tolerance ``tol`` defaults to 3/4 of the working precision.
    Lowering the tolerance is useful for finding approximate matches.
    We can for example try to generate approximations for pi::

        >>> mp.dps = 15
        >>> identify(pi, tol=1e-2)
        '(22/7)'
        >>> identify(pi, tol=1e-3)
        '(355/113)'
        >>> identify(pi, tol=1e-10)
        '(5**(339/269))/(2**(64/269)*3**(13/269)*7**(92/269))'

    With ``full=True``, and by supplying a few base constants,
    ``identify`` can generate almost endless lists of approximations
    for any number (the output below has been truncated to show only
    the first few)::

        >>> for p in identify(pi, ['e', 'catalan'], tol=1e-5, full=True):
        ...     print(p)
        ...  # doctest: +ELLIPSIS
        e/log((6 + (-4/3)*e))
        (3**3*5*e*catalan**2)/(2*7**2)
        sqrt(((-13) + 1*e + 22*catalan))
        log(((-6) + 24*e + 4*catalan)/e)
        exp(catalan*((-1/5) + (8/15)*e))
        catalan*(6 + (-6)*e + 15*catalan)
        sqrt((5 + 26*e + (-3)*catalan))/e
        e*sqrt(((-27) + 2*e + 25*catalan))
        log(((-1) + (-11)*e + 59*catalan))
        ((3/20) + (21/20)*e + (3/20)*catalan)
        ...

    The numerical values are roughly as close to `\pi` as permitted by the
    specified tolerance:

        >>> e/log(6-4*e/3)
        3.14157719846001
        >>> 135*e*catalan**2/98
        3.14166950419369
        >>> sqrt(e-13+22*catalan)
        3.14158000062992
        >>> log(24*e-6+4*catalan)-1
        3.14158791577159

    **Symbolic processing**

    The output formula can be evaluated as a Python expression.
    Note however that if fractions (like '2/3') are present in
    the formula, Python's :func:`~mpmath.eval()` may erroneously perform
    integer division. Note also that the output is not necessarily
    in the algebraically simplest form::

        >>> identify(sqrt(2))
        '(sqrt(8)/2)'

    As a solution to both problems, consider using SymPy's
    :func:`~mpmath.sympify` to convert the formula into a symbolic expression.
    SymPy can be used to pretty-print or further simplify the formula
    symbolically::

        >>> from sympy import sympify # doctest: +SKIP
        >>> sympify(identify(sqrt(2))) # doctest: +SKIP
        2**(1/2)

    Sometimes :func:`~mpmath.identify` can simplify an expression further than
    a symbolic algorithm::

        >>> from sympy import simplify # doctest: +SKIP
        >>> x = sympify('-1/(-3/2+(1/2)*5**(1/2))*(3/2-1/2*5**(1/2))**(1/2)') # doctest: +SKIP
        >>> x # doctest: +SKIP
        (3/2 - 5**(1/2)/2)**(-1/2)
        >>> x = simplify(x) # doctest: +SKIP
        >>> x # doctest: +SKIP
        2/(6 - 2*5**(1/2))**(1/2)
        >>> mp.dps = 30 # doctest: +SKIP
        >>> x = sympify(identify(x.evalf(30))) # doctest: +SKIP
        >>> x # doctest: +SKIP
        1/2 + 5**(1/2)/2

    (In fact, this functionality is available directly in SymPy as the
    function :func:`~mpmath.nsimplify`, which is essentially a wrapper for
    :func:`~mpmath.identify`.)

    **Miscellaneous issues and limitations**

    The input `x` must be a real number. All base constants must be
    positive real numbers and must not be rationals or rational linear
    combinations of each other.

    The worst-case computation time grows quickly with the number of
    base constants. Already with 3 or 4 base constants,
    :func:`~mpmath.identify` may require several seconds to finish. To search
    for relations among a large number of constants, you should
    consider using :func:`~mpmath.pslq` directly.

    The extended transformations are applied to x, not the constants
    separately. As a result, ``identify`` will for example be able to
    recognize ``exp(2*pi+3)`` with ``pi`` given as a base constant, but
    not ``2*exp(pi)+3``. It will be able to recognize the latter if
    ``exp(pi)`` is given explicitly as a base constant.

    