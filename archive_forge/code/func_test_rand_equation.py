import numpy as np
import pytest
from opt_einsum import contract, contract_expression, contract_path, helpers
from opt_einsum.paths import linear_to_ssa, ssa_to_linear
@pytest.mark.parametrize('optimize', ['greedy', 'optimal'])
@pytest.mark.parametrize('n', [4, 5])
@pytest.mark.parametrize('reg', [2, 3])
@pytest.mark.parametrize('n_out', [0, 2, 4])
@pytest.mark.parametrize('global_dim', [False, True])
def test_rand_equation(optimize, n, reg, n_out, global_dim):
    eq, _, size_dict = helpers.rand_equation(n, reg, n_out, d_min=2, d_max=5, seed=42, return_size_dict=True)
    views = helpers.build_views(eq, size_dict)
    expected = contract(eq, *views, optimize=False)
    actual = contract(eq, *views, optimize=optimize)
    assert np.allclose(expected, actual)