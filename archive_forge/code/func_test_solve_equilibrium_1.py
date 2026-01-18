from ..util.testing import requires
from .._equilibrium import equilibrium_residual, solve_equilibrium, _get_rc_interval
from .._util import prodpow
@requires('numpy')
def test_solve_equilibrium_1():
    c = np.array((13.0, 11, 17))
    stoich = np.array((-2, 3, -4))
    K = 3.14

    def f(x):
        return (13 - 2 * x) ** (-2) * (11 + 3 * x) ** 3 * (17 - 4 * x) ** (-4) - K
    assert np.allclose(solve_equilibrium(c, stoich, K), c + stoich * fsolve(f, 3.48))