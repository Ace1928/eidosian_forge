from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _interpolate_multivariate(evalpoints, hpeval, ring, i, p, ground=False):
    """
    Reconstruct a polynomial `h_p` in `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]`
    from a list of evaluation points in `\\mathbb{Z}_p` and a list of
    polynomials in
    `\\mathbb{Z}_p[x_0, \\ldots, x_{i-1}, x_{i+1}, \\ldots, x_{k-1}]`, which
    are the images of `h_p` evaluated in the variable `x_i`.

    It is also possible to reconstruct a parameter of the ground domain,
    i.e. if `h_p` is a polynomial over `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]`.
    In this case, one has to set ``ground=True``.

    Parameters
    ==========

    evalpoints : list of Integer objects
        list of evaluation points in `\\mathbb{Z}_p`
    hpeval : list of PolyElement objects
        list of polynomials in (resp. over)
        `\\mathbb{Z}_p[x_0, \\ldots, x_{i-1}, x_{i+1}, \\ldots, x_{k-1}]`,
        images of `h_p` evaluated in the variable `x_i`
    ring : PolyRing
        `h_p` will be an element of this ring
    i : Integer
        index of the variable which has to be reconstructed
    p : Integer
        prime number, modulus of `h_p`
    ground : Boolean
        indicates whether `x_i` is in the ground domain, default is
        ``False``

    Returns
    =======

    hp : PolyElement
        interpolated polynomial in (resp. over)
        `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]`

    """
    hp = ring.zero
    if ground:
        domain = ring.domain.domain
        y = ring.domain.gens[i]
    else:
        domain = ring.domain
        y = ring.gens[i]
    for a, hpa in zip(evalpoints, hpeval):
        numer = ring.one
        denom = domain.one
        for b in evalpoints:
            if b == a:
                continue
            numer *= y - b
            denom *= a - b
        denom = domain.invert(denom, p)
        coeff = numer.mul_ground(denom)
        hp += hpa.set_ring(ring) * coeff
    return hp.trunc_ground(p)