import os
from ....context import cpu
from ...block import HybridBlock
from ... import nn
from ...contrib.nn import HybridConcurrent
from .... import base
def _make_A(pool_features, prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, (64, 1, None, None)))
        out.add(_make_branch(None, (48, 1, None, None), (64, 5, None, 2)))
        out.add(_make_branch(None, (64, 1, None, None), (96, 3, None, 1), (96, 3, None, 1)))
        out.add(_make_branch('avg', (pool_features, 1, None, None)))
    return out