import numpy as np
import pytest
from opt_einsum import (backends, contract, contract_expression, helpers, sharing)
from opt_einsum.contract import Shaped, infer_backend, parse_backend
@pytest.mark.parametrize('string', tests)
def test_object_arrays_backend(string):
    views = helpers.build_views(string)
    ein = contract(string, *views, optimize=False, use_blas=False)
    assert ein.dtype != object
    shps = [v.shape for v in views]
    expr = contract_expression(string, *shps, optimize=True)
    obj_views = [view.astype(object) for view in views]
    obj_opt = contract(string, *obj_views, backend='object')
    assert obj_opt.dtype == object
    assert np.allclose(ein, obj_opt.astype(float))
    obj_opt = expr(*obj_views, backend='object')
    assert obj_opt.dtype == object
    assert np.allclose(ein, obj_opt.astype(float))