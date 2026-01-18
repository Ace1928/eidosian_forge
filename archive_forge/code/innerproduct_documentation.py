from sympy.core.expr import Expr
from sympy.functions.elementary.complexes import conjugate
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.state import KetBase, BraBase
An unevaluated inner product between a Bra and a Ket [1].

    Parameters
    ==========

    bra : BraBase or subclass
        The bra on the left side of the inner product.
    ket : KetBase or subclass
        The ket on the right side of the inner product.

    Examples
    ========

    Create an InnerProduct and check its properties:

        >>> from sympy.physics.quantum import Bra, Ket
        >>> b = Bra('b')
        >>> k = Ket('k')
        >>> ip = b*k
        >>> ip
        <b|k>
        >>> ip.bra
        <b|
        >>> ip.ket
        |k>

    In simple products of kets and bras inner products will be automatically
    identified and created::

        >>> b*k
        <b|k>

    But in more complex expressions, there is ambiguity in whether inner or
    outer products should be created::

        >>> k*b*k*b
        |k><b|*|k>*<b|

    A user can force the creation of a inner products in a complex expression
    by using parentheses to group the bra and ket::

        >>> k*(b*k)*b
        <b|k>*|k>*<b|

    Notice how the inner product <b|k> moved to the left of the expression
    because inner products are commutative complex numbers.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inner_product
    