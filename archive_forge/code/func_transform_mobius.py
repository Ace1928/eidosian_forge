import sys
import os
from six import iteritems
from enum import IntEnum
from contextlib import contextmanager
import json
def transform_mobius(pool):
    """
    Transform pool for fast dot product search on HNSW graph
    https://papers.nips.cc/paper/9032-mobius-transformation-for-fast-inner-product-search-on-graph.pdf

    Parameters
    ----------
    pool : Pool

    Returns
    -------
    transformed_pool : Pool
    """
    transformed_pool = Pool.from_bytes(bytes(0), EVectorComponentType.Float, pool.dimension)
    transformed_pool._storage = _transform_mobius[pool.dtype](pool._storage)
    return transformed_pool