from sympy.polys.domains.domainelement import DomainElement
from sympy.utilities import public
from mpmath.ctx_mp_python import PythonMPContext, _mpf, _mpc, _constant
from mpmath.libmp import (MPZ_ONE, fzero, fone, finf, fninf, fnan,
from mpmath.rational import mpq
def to_rational(ctx, s, limit=True):
    p, q = to_rational(s._mpf_)
    if not limit or q <= ctx.max_denom:
        return (p, q)
    p0, q0, p1, q1 = (0, 1, 1, 0)
    n, d = (p, q)
    while True:
        a = n // d
        q2 = q0 + a * q1
        if q2 > ctx.max_denom:
            break
        p0, q0, p1, q1 = (p1, q1, p0 + a * p1, q2)
        n, d = (d, n - a * d)
    k = (ctx.max_denom - q0) // q1
    number = mpq(p, q)
    bound1 = mpq(p0 + k * p1, q0 + k * q1)
    bound2 = mpq(p1, q1)
    if not bound2 or not bound1:
        return (p, q)
    elif abs(bound2 - number) <= abs(bound1 - number):
        return bound2._mpq_
    else:
        return bound1._mpq_