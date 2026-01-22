from sympy.polys.polytools import Poly
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.utilities.decorator import public
from .modules import ModuleEndomorphism, ModuleHomomorphism, PowerBasis
from .utilities import extract_fundamental_discriminant

    Zassenhaus's "Round 2" algorithm.

    Explanation
    ===========

    Carry out Zassenhaus's "Round 2" algorithm on an irreducible polynomial
    *T* over :ref:`ZZ` or :ref:`QQ`. This computes an integral basis and the
    discriminant for the field $K = \mathbb{Q}[x]/(T(x))$.

    Alternatively, you may pass an :py:class:`~.AlgebraicField` instance, in
    place of the polynomial *T*, in which case the algorithm is applied to the
    minimal polynomial for the field's primitive element.

    Ordinarily this function need not be called directly, as one can instead
    access the :py:meth:`~.AlgebraicField.maximal_order`,
    :py:meth:`~.AlgebraicField.integral_basis`, and
    :py:meth:`~.AlgebraicField.discriminant` methods of an
    :py:class:`~.AlgebraicField`.

    Examples
    ========

    Working through an AlgebraicField:

    >>> from sympy import Poly, QQ
    >>> from sympy.abc import x
    >>> T = Poly(x ** 3 + x ** 2 - 2 * x + 8)
    >>> K = QQ.alg_field_from_poly(T, "theta")
    >>> print(K.maximal_order())
    Submodule[[2, 0, 0], [0, 2, 0], [0, 1, 1]]/2
    >>> print(K.discriminant())
    -503
    >>> print(K.integral_basis(fmt='sympy'))
    [1, theta, theta/2 + theta**2/2]

    Calling directly:

    >>> from sympy import Poly
    >>> from sympy.abc import x
    >>> from sympy.polys.numberfields.basis import round_two
    >>> T = Poly(x ** 3 + x ** 2 - 2 * x + 8)
    >>> print(round_two(T))
    (Submodule[[2, 0, 0], [0, 2, 0], [0, 1, 1]]/2, -503)

    The nilradicals mod $p$ that are sometimes computed during the Round Two
    algorithm may be useful in further calculations. Pass a dictionary under
    `radicals` to receive these:

    >>> T = Poly(x**3 + 3*x**2 + 5)
    >>> rad = {}
    >>> ZK, dK = round_two(T, radicals=rad)
    >>> print(rad)
    {3: Submodule[[-1, 1, 0], [-1, 0, 1]]}

    Parameters
    ==========

    T : :py:class:`~.Poly`, :py:class:`~.AlgebraicField`
        Either (1) the irreducible polynomial over :ref:`ZZ` or :ref:`QQ`
        defining the number field, or (2) an :py:class:`~.AlgebraicField`
        representing the number field itself.

    radicals : dict, optional
        This is a way for any $p$-radicals (if computed) to be returned by
        reference. If desired, pass an empty dictionary. If the algorithm
        reaches the point where it computes the nilradical mod $p$ of the ring
        of integers $Z_K$, then an $\mathbb{F}_p$-basis for this ideal will be
        stored in this dictionary under the key ``p``. This can be useful for
        other algorithms, such as prime decomposition.

    Returns
    =======

    Pair ``(ZK, dK)``, where:

        ``ZK`` is a :py:class:`~sympy.polys.numberfields.modules.Submodule`
        representing the maximal order.

        ``dK`` is the discriminant of the field $K = \mathbb{Q}[x]/(T(x))$.

    See Also
    ========

    .AlgebraicField.maximal_order
    .AlgebraicField.integral_basis
    .AlgebraicField.discriminant

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*

    