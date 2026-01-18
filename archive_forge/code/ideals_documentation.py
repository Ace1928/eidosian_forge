from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable

        Express ``e`` in terms of the generators of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x**2 + 1, x)
        >>> I.in_terms_of_generators(1)
        [1, -x]
        