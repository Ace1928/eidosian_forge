import numpy as np
import pytest
from opt_einsum import (backends, contract, contract_expression, helpers, sharing)
from opt_einsum.contract import Shaped, infer_backend, parse_backend
@pytest.mark.skipif(not found_theano, reason='Theano not installed.')
@pytest.mark.parametrize('string', tests)
def test_theano(string):
    views = helpers.build_views(string)
    ein = contract(string, *views, optimize=False, use_blas=False)
    shps = [v.shape for v in views]
    expr = contract_expression(string, *shps, optimize=True)
    opt = expr(*views, backend='theano')
    assert np.allclose(ein, opt)
    theano_views = [backends.to_theano(view) for view in views]
    theano_opt = expr(*theano_views)
    assert isinstance(theano_opt, theano.tensor.TensorVariable)