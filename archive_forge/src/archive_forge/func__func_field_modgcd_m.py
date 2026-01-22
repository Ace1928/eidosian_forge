from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _func_field_modgcd_m(f, g, minpoly):
    """
    Compute the GCD of two polynomials in
    `\\mathbb Q(t_1, \\ldots, t_k)[z]/(m_{\\alpha}(z))[x]` using a modular
    algorithm.

    The algorithm computes the GCD of two polynomials `f` and `g` by
    calculating the GCD in
    `\\mathbb Z_p(t_1, \\ldots, t_k)[z] / (\\check m_{\\alpha}(z))[x]` for
    suitable primes `p` and the primitive associate `\\check m_{\\alpha}(z)`
    of `m_{\\alpha}(z)`. Then the coefficients are reconstructed with the
    Chinese Remainder Theorem and Rational Reconstruction. To compute the
    GCD over `\\mathbb Z_p(t_1, \\ldots, t_k)[z] / (\\check m_{\\alpha})[x]`,
    the recursive subroutine ``_func_field_modgcd_p`` is used. To verify the
    result in `\\mathbb Q(t_1, \\ldots, t_k)[z] / (m_{\\alpha}(z))[x]`, a
    fraction free trial division is used.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\\mathbb Z[t_1, \\ldots, t_k][x, z]`
    minpoly : PolyElement
        irreducible polynomial in `\\mathbb Z[t_1, \\ldots, t_k][z]`

    Returns
    =======

    h : PolyElement
        the primitive associate in `\\mathbb Z[t_1, \\ldots, t_k][x, z]` of
        the GCD of `f` and `g`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _func_field_modgcd_m
    >>> from sympy.polys import ring, ZZ

    >>> R, x, z = ring('x, z', ZZ)
    >>> minpoly = (z**2 - 2).drop(0)

    >>> f = x**2 + 2*x*z + 2
    >>> g = x + z
    >>> _func_field_modgcd_m(f, g, minpoly)
    x + z

    >>> D, t = ring('t', ZZ)
    >>> R, x, z = ring('x, z', D)
    >>> minpoly = (z**2-3).drop(0)

    >>> f = x**2 + (t + 1)*x*z + 3*t
    >>> g = x*z + 3*t
    >>> _func_field_modgcd_m(f, g, minpoly)
    x + t*z

    References
    ==========

    1. [Hoeij04]_

    See also
    ========

    _func_field_modgcd_p

    """
    ring = f.ring
    domain = ring.domain
    if isinstance(domain, PolynomialRing):
        k = domain.ngens
        QQdomain = domain.ring.clone(domain=domain.domain.get_field())
        QQring = ring.clone(domain=QQdomain)
    else:
        k = 0
        QQring = ring.clone(domain=ring.domain.get_field())
    cf, f = f.primitive()
    cg, g = g.primitive()
    gamma = ring.dmp_LC(f) * ring.dmp_LC(g)
    delta = minpoly.LC
    p = 1
    primes = []
    hplist = []
    LMlist = []
    while True:
        p = nextprime(p)
        if gamma.trunc_ground(p) == 0:
            continue
        if k == 0:
            test = delta % p == 0
        else:
            test = delta.trunc_ground(p) == 0
        if test:
            continue
        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)
        minpolyp = minpoly.trunc_ground(p)
        hp = _func_field_modgcd_p(fp, gp, minpolyp, p)
        if hp is None:
            continue
        if hp == 1:
            return ring.one
        LM = [hp.degree()] + [0] * k
        if k > 0:
            for monom, coeff in hp.iterterms():
                if monom[0] == LM[0] and coeff.LM > tuple(LM[1:]):
                    LM[1:] = coeff.LM
        hm = hp
        m = p
        for q, hq, LMhq in zip(primes, hplist, LMlist):
            if LMhq == LM:
                hm = _chinese_remainder_reconstruction_multivariate(hq, hm, q, m)
                m *= q
        primes.append(p)
        hplist.append(hp)
        LMlist.append(LM)
        hm = _rational_reconstruction_int_coeffs(hm, m, QQring)
        if hm is None:
            continue
        if k == 0:
            h = hm.clear_denoms()[1]
        else:
            den = domain.domain.one
            for coeff in hm.itercoeffs():
                den = domain.domain.lcm(den, coeff.clear_denoms()[0])
            h = hm.mul_ground(den)
        h = h.set_ring(ring)
        h = h.primitive()[1]
        if not (_trial_division(f.mul_ground(cf), h, minpoly) or _trial_division(g.mul_ground(cg), h, minpoly)):
            return h