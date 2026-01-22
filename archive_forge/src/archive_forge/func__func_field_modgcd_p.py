from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _func_field_modgcd_p(f, g, minpoly, p):
    """
    Compute the GCD of two polynomials `f` and `g` in
    `\\mathbb Z_p(t_1, \\ldots, t_k)[z]/(\\check m_\\alpha(z))[x]`.

    The algorithm reduces the problem step by step by evaluating the
    polynomials `f` and `g` at `t_k = a` for suitable `a \\in \\mathbb Z_p`
    and then calls itself recursively to compute the GCD in
    `\\mathbb Z_p(t_1, \\ldots, t_{k-1})[z]/(\\check m_\\alpha(z))[x]`. If these
    recursive calls are successful, the GCD over `k` variables is
    interpolated, otherwise the algorithm returns ``None``. After
    interpolation, Rational Function Reconstruction is used to obtain the
    correct coefficients. If this fails, a new evaluation point has to be
    chosen, otherwise the desired polynomial is obtained by clearing
    denominators. The result is verified with a fraction free trial
    division.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\\mathbb Z[t_1, \\ldots, t_k][x, z]`
    minpoly : PolyElement
        polynomial in `\\mathbb Z[t_1, \\ldots, t_k][z]`, not necessarily
        irreducible
    p : Integer
        prime number, modulus of `\\mathbb Z_p`

    Returns
    =======

    h : PolyElement
        primitive associate in `\\mathbb Z[t_1, \\ldots, t_k][x, z]` of the
        GCD of the polynomials `f` and `g`  or ``None``, coefficients are
        in `\\left[ -\\frac{p-1} 2, \\frac{p-1} 2 \\right]`

    References
    ==========

    1. [Hoeij04]_

    """
    ring = f.ring
    domain = ring.domain
    if isinstance(domain, PolynomialRing):
        k = domain.ngens
    else:
        return _euclidean_algorithm(f, g, minpoly, p)
    if k == 1:
        qdomain = domain.ring.to_field()
    else:
        qdomain = domain.ring.drop_to_ground(k - 1)
        qdomain = qdomain.clone(domain=qdomain.domain.ring.to_field())
    qring = ring.clone(domain=qdomain)
    n = 1
    d = 1
    gamma = ring.dmp_LC(f) * ring.dmp_LC(g)
    delta = minpoly.LC
    evalpoints = []
    heval = []
    LMlist = []
    points = list(range(p))
    while points:
        a = random.sample(points, 1)[0]
        points.remove(a)
        if k == 1:
            test = delta.evaluate(k - 1, a) % p == 0
        else:
            test = delta.evaluate(k - 1, a).trunc_ground(p) == 0
        if test:
            continue
        gammaa = _evaluate_ground(gamma, k - 1, a)
        minpolya = _evaluate_ground(minpoly, k - 1, a)
        if gammaa.rem([minpolya, gammaa.ring(p)]) == 0:
            continue
        fa = _evaluate_ground(f, k - 1, a)
        ga = _evaluate_ground(g, k - 1, a)
        ha = _func_field_modgcd_p(fa, ga, minpolya, p)
        if ha is None:
            d += 1
            if d > n:
                return None
            continue
        if ha == 1:
            return ha
        LM = [ha.degree()] + [0] * (k - 1)
        if k > 1:
            for monom, coeff in ha.iterterms():
                if monom[0] == LM[0] and coeff.LM > tuple(LM[1:]):
                    LM[1:] = coeff.LM
        evalpoints_a = [a]
        heval_a = [ha]
        if k == 1:
            m = qring.domain.get_ring().one
        else:
            m = qring.domain.domain.get_ring().one
        t = m.ring.gens[0]
        for b, hb, LMhb in zip(evalpoints, heval, LMlist):
            if LMhb == LM:
                evalpoints_a.append(b)
                heval_a.append(hb)
                m *= t - b
        m = m.trunc_ground(p)
        evalpoints.append(a)
        heval.append(ha)
        LMlist.append(LM)
        n += 1
        h = _interpolate_multivariate(evalpoints_a, heval_a, ring, k - 1, p, ground=True)
        h = _rational_reconstruction_func_coeffs(h, p, m, qring, k - 1)
        if h is None:
            continue
        if k == 1:
            dom = qring.domain.field
            den = dom.ring.one
            for coeff in h.itercoeffs():
                den = dom.ring.from_dense(gf_lcm(den.to_dense(), coeff.denom.to_dense(), p, dom.domain))
        else:
            dom = qring.domain.domain.field
            den = dom.ring.one
            for coeff in h.itercoeffs():
                for c in coeff.itercoeffs():
                    den = dom.ring.from_dense(gf_lcm(den.to_dense(), c.denom.to_dense(), p, dom.domain))
        den = qring.domain_new(den.trunc_ground(p))
        h = ring(h.mul_ground(den).as_expr()).trunc_ground(p)
        if not _trial_division(f, h, minpoly, p) and (not _trial_division(g, h, minpoly, p)):
            return h
    return None