import itertools
import weakref
from collections import Counter
import numpy as np
import pytest
from opt_einsum import (contract, contract_expression, contract_path, get_symbol, helpers, shared_intermediates)
from opt_einsum.backends import to_cupy, to_torch
from opt_einsum.contract import _einsum
from opt_einsum.parser import parse_einsum_input
from opt_einsum.sharing import (count_cached_ops, currently_sharing, get_sharing_cache)
@pytest.mark.parametrize('backend', backends)
def test_partial_sharing(backend):
    eq = 'ab,bc,de->'
    x, y, z1 = helpers.build_views(eq)
    z2 = 2.0 * z1 - 1.0
    expr = contract_expression(eq, x.shape, y.shape, z1.shape)
    print('-' * 40)
    print('Without sharing:')
    num_exprs_nosharing = Counter()
    with shared_intermediates() as cache:
        expr(x, y, z1, backend=backend)
        num_exprs_nosharing.update(count_cached_ops(cache))
    with shared_intermediates() as cache:
        expr(x, y, z2, backend=backend)
        num_exprs_nosharing.update(count_cached_ops(cache))
    print('-' * 40)
    print('With sharing:')
    with shared_intermediates() as cache:
        expr(x, y, z1, backend=backend)
        expr(x, y, z2, backend=backend)
        num_exprs_sharing = count_cached_ops(cache)
    print('-' * 40)
    print('Without sharing: {} expressions'.format(num_exprs_nosharing))
    print('With sharing: {} expressions'.format(num_exprs_sharing))
    assert num_exprs_nosharing['einsum'] > num_exprs_sharing['einsum']