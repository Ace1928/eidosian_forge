from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.functions import (arg, atan2, bernoulli, beta, ceiling, chebyshevu,
from sympy.functions import (sin, cos, tan, cot, sec, csc, asin, acos, acot,
from sympy.testing.pytest import raises, XFAIL
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
from sympy.functions.special.bessel import (jn, yn, besselj, bessely, besseli,
from sympy.functions.special.gamma_functions import (gamma, lowergamma,
from sympy.functions.special.error_functions import (Chi, Ci, erf, erfc, erfi,
from sympy.printing.octave import octave_code, octave_code as mcode
def test_specfun():
    n = Symbol('n')
    for f in [besselj, bessely, besseli, besselk]:
        assert octave_code(f(n, x)) == f.__name__ + '(n, x)'
    for f in (erfc, erfi, erf, erfinv, erfcinv, fresnelc, fresnels, gamma):
        assert octave_code(f(x)) == f.__name__ + '(x)'
    assert octave_code(hankel1(n, x)) == 'besselh(n, 1, x)'
    assert octave_code(hankel2(n, x)) == 'besselh(n, 2, x)'
    assert octave_code(airyai(x)) == 'airy(0, x)'
    assert octave_code(airyaiprime(x)) == 'airy(1, x)'
    assert octave_code(airybi(x)) == 'airy(2, x)'
    assert octave_code(airybiprime(x)) == 'airy(3, x)'
    assert octave_code(uppergamma(n, x)) == "(gammainc(x, n, 'upper').*gamma(n))"
    assert octave_code(lowergamma(n, x)) == '(gammainc(x, n).*gamma(n))'
    assert octave_code(z ** lowergamma(n, x)) == 'z.^(gammainc(x, n).*gamma(n))'
    assert octave_code(jn(n, x)) == 'sqrt(2)*sqrt(pi)*sqrt(1./x).*besselj(n + 1/2, x)/2'
    assert octave_code(yn(n, x)) == 'sqrt(2)*sqrt(pi)*sqrt(1./x).*bessely(n + 1/2, x)/2'
    assert octave_code(LambertW(x)) == 'lambertw(x)'
    assert octave_code(LambertW(x, n)) == 'lambertw(n, x)'
    assert octave_code(Ei(x)) == 'logint(exp(x))'
    assert octave_code(dirichlet_eta(x)) == '((x == 1).*(log(2)) + (~(x == 1)).*((1 - 2.^(1 - x)).*zeta(x)))'
    assert octave_code(riemann_xi(x)) == 'pi.^(-x/2).*x.*(x - 1).*gamma(x/2).*zeta(x)/2'