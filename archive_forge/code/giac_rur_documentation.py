from sage.all import QQ, PolynomialRing, NumberField
from .giac_helper import giac

    Suppose an ideal I in QQ[x_1,...,x_n] is 0 dimensional, and we
    want to describe all the points of the finite set V(I) in CC^n.  A
    rational univariate representation (RUR) of V(I), is a collection
    of univariate polynomials h, g_0, g_1, ... , g_n in QQ[t] where
    deg(h) = #V(I) and deg(g_i) < deg(h) such that the points of V(I)
    correspond precisely to

       (g_1(z)/g_0(z), (g_2(z)/g_0(z), ... (g_n(z)/g_0(z))

    where z in CC is a root of h.

    In this variant, we factor h into irreducibles return each part of
    the RUR individually.

    Example:

    sage: R = PolynomialRing(QQ, ['x', 'y', 'z'])
    sage: x, y, z = R.gens()
    sage: I = R.ideal([x + y + z*z, x*y*z - 3, x*x + y*y + z*z - 2])
    sage: ans = rational_univariate_representation(I)
    sage: len(ans)
    1
    sage: K, rep, mult = ans[0]
    sage: mult
    1
    sage: h = K.polynomial(); h
    x^10 - 2*x^9 - 4*x^8 + 6*x^7 + 7*x^6 - 13*x^5 - 17/2*x^4 + 36*x^3 + 63/2*x^2 + 81/2
    sage: rep[y]
    a
    sage: 1215 * rep[x]  # Here g0 = 1215
    8*a^9 + 8*a^8 - 8*a^7 - 246*a^6 + 128*a^5 + 550*a^4 - 308*a^3 - 636*a^2 + 639*a + 1917
    sage: I.subs(rep).is_zero()
    True

    Here is an example using a Ptolemy variety:

    sage: M = Manifold('t00000')
    sage: obs = M.ptolemy_generalized_obstruction_classes(2)[1]
    sage: V = M.ptolemy_variety(2, obs)
    sage: I = V.ideal_with_non_zero_condition
    sage: ans = rational_univariate_representation(I)
    sage: ans[0][0].polynomial()
    x^8 - 4*x^7 - 2*x^6 + 14*x^5 + 14*x^4 - 7*x^3 - 13*x^2 - x + 5

    For more, see:

    https://en.wikipedia.org/wiki/System_of_polynomial_equations#Rational_univariate_representation

    