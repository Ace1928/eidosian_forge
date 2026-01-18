import numpy as np
import pytest
from opt_einsum import (backends, contract, contract_expression, helpers, sharing)
from opt_einsum.contract import Shaped, infer_backend, parse_backend
@pytest.mark.skipif(not found_torch, reason='Torch not installed.')
@pytest.mark.parametrize('constants', [{0, 1}, {0, 2}, {1, 2}])
def test_torch_with_constants(constants):
    eq = 'ij,jk,kl->li'
    shapes = ((2, 3), (3, 4), (4, 5))
    non_const, = {0, 1, 2} - constants
    ops = [np.random.rand(*shp) if i in constants else shp for i, shp in enumerate(shapes)]
    var = np.random.rand(*shapes[non_const])
    res_exp = contract(eq, *(ops[i] if i in constants else var for i in range(3)))
    expr = contract_expression(eq, *ops, constants=constants)
    res_got = expr(var, backend='torch')
    assert all((array is None or infer_backend(array) == 'torch' for array in expr._evaluated_constants['torch']))
    assert np.allclose(res_exp, res_got)
    res_got2 = expr(var, backend='numpy')
    assert np.allclose(res_exp, res_got2)
    res_got3 = expr(backends.to_torch(var))
    assert isinstance(res_got3, torch.Tensor)
    res_got3 = res_got3.numpy() if res_got3.device.type == 'cpu' else res_got3.cpu().numpy()
    assert np.allclose(res_exp, res_got3)