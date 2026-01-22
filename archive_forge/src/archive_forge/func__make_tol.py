from sympy.polys.domains.domainelement import DomainElement
from sympy.utilities import public
from mpmath.ctx_mp_python import PythonMPContext, _mpf, _mpc, _constant
from mpmath.libmp import (MPZ_ONE, fzero, fone, finf, fninf, fnan,
from mpmath.rational import mpq
def _make_tol(ctx):
    hundred = (0, 25, 2, 5)
    eps = (0, MPZ_ONE, 1 - ctx.prec, 1)
    return mpf_mul(hundred, eps)