from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _modgcd_multivariate_p(f, g, p, degbound, contbound):
    """
    Compute the GCD of two polynomials in
    `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]`.

    The algorithm reduces the problem step by step by evaluating the
    polynomials `f` and `g` at `x_{k-1} = a` for suitable
    `a \\in \\mathbb{Z}_p` and then calls itself recursively to compute the GCD
    in `\\mathbb{Z}_p[x_0, \\ldots, x_{k-2}]`. If these recursive calls are
    successful for enough evaluation points, the GCD in `k` variables is
    interpolated, otherwise the algorithm returns ``None``. Every time a GCD
    or a content is computed, their degrees are compared with the bounds. If
    a degree greater then the bound is encountered, then the current call
    returns ``None`` and a new evaluation point has to be chosen. If at some
    point the degree is smaller, the correspondent bound is updated and the
    algorithm fails.

    Parameters
    ==========

    f : PolyElement
        multivariate integer polynomial with coefficients in `\\mathbb{Z}_p`
    g : PolyElement
        multivariate integer polynomial with coefficients in `\\mathbb{Z}_p`
    p : Integer
        prime number, modulus of `f` and `g`
    degbound : list of Integer objects
        ``degbound[i]`` is an upper bound for the degree of the GCD of `f`
        and `g` in the variable `x_i`
    contbound : list of Integer objects
        ``contbound[i]`` is an upper bound for the degree of the content of
        the GCD in `\\mathbb{Z}_p[x_i][x_0, \\ldots, x_{i-1}]`,
        ``contbound[0]`` is not used can therefore be chosen
        arbitrarily.

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g` or ``None``

    References
    ==========

    1. [Monagan00]_
    2. [Brown71]_

    """
    ring = f.ring
    k = ring.ngens
    if k == 1:
        h = _gf_gcd(f, g, p).trunc_ground(p)
        degh = h.degree()
        if degh > degbound[0]:
            return None
        if degh < degbound[0]:
            degbound[0] = degh
            raise ModularGCDFailed
        return h
    degyf = f.degree(k - 1)
    degyg = g.degree(k - 1)
    contf, f = _primitive(f, p)
    contg, g = _primitive(g, p)
    conth = _gf_gcd(contf, contg, p)
    degcontf = contf.degree()
    degcontg = contg.degree()
    degconth = conth.degree()
    if degconth > contbound[k - 1]:
        return None
    if degconth < contbound[k - 1]:
        contbound[k - 1] = degconth
        raise ModularGCDFailed
    lcf = _LC(f)
    lcg = _LC(g)
    delta = _gf_gcd(lcf, lcg, p)
    evaltest = delta
    for i in range(k - 1):
        evaltest *= _gf_gcd(_LC(_swap(f, i)), _LC(_swap(g, i)), p)
    degdelta = delta.degree()
    N = min(degyf - degcontf, degyg - degcontg, degbound[k - 1] - contbound[k - 1] + degdelta) + 1
    if p < N:
        return None
    n = 0
    d = 0
    evalpoints = []
    heval = []
    points = list(range(p))
    while points:
        a = random.sample(points, 1)[0]
        points.remove(a)
        if not evaltest.evaluate(0, a) % p:
            continue
        deltaa = delta.evaluate(0, a) % p
        fa = f.evaluate(k - 1, a).trunc_ground(p)
        ga = g.evaluate(k - 1, a).trunc_ground(p)
        ha = _modgcd_multivariate_p(fa, ga, p, degbound, contbound)
        if ha is None:
            d += 1
            if d > n:
                return None
            continue
        if ha.is_ground:
            h = conth.set_ring(ring).trunc_ground(p)
            return h
        ha = ha.mul_ground(deltaa).trunc_ground(p)
        evalpoints.append(a)
        heval.append(ha)
        n += 1
        if n == N:
            h = _interpolate_multivariate(evalpoints, heval, ring, k - 1, p)
            h = _primitive(h, p)[1] * conth.set_ring(ring)
            degyh = h.degree(k - 1)
            if degyh > degbound[k - 1]:
                return None
            if degyh < degbound[k - 1]:
                degbound[k - 1] = degyh
                raise ModularGCDFailed
            return h
    return None