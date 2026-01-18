import numpy as np
import pytest
from opt_einsum import (backends, contract, contract_expression, helpers, sharing)
from opt_einsum.contract import Shaped, infer_backend, parse_backend
@pytest.mark.skipif(not found_tensorflow, reason='Tensorflow not installed.')
@pytest.mark.parametrize('string', tests)
def test_tensorflow_with_sharing(string):
    views = helpers.build_views(string)
    ein = contract(string, *views, optimize=False, use_blas=False)
    shps = [v.shape for v in views]
    expr = contract_expression(string, *shps, optimize=True)
    sess = tf.Session(config=_TF_CONFIG)
    with sess.as_default(), sharing.shared_intermediates() as cache:
        tfl1 = expr(*views, backend='tensorflow')
        assert sharing.get_sharing_cache() is cache
        cache_sz = len(cache)
        assert cache_sz > 0
        tfl2 = expr(*views, backend='tensorflow')
        assert len(cache) == cache_sz
    assert all((isinstance(t, tf.Tensor) for t in cache.values()))
    assert np.allclose(ein, tfl1)
    assert np.allclose(ein, tfl2)