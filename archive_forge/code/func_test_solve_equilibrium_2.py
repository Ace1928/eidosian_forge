from ..util.testing import requires
from .._equilibrium import equilibrium_residual, solve_equilibrium, _get_rc_interval
from .._util import prodpow
@requires('numpy')
def test_solve_equilibrium_2():
    c = np.array([0.0017, 3000000.0, 3000000.0, 97000000.0, 5550000000.0])
    stoich = (1, 1, 0, 0, -1)
    K = 55 * 1e-06

    def f(x):
        return prodpow(c + x * stoich, stoich) - K
    solution = solve_equilibrium(c, stoich, K)
    assert np.allclose(solution, c + stoich * fsolve(f, 0.1))