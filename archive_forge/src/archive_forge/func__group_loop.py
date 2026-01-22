from .truncatedComplex import TruncatedComplex
from snappy.snap import t3mlite as t3m
from .verificationError import *
def _group_loop(loop):
    indices = [i for i, edge in enumerate(loop) if edge.subcomplex_type == 'edgeLoop'] + [len(loop)]
    if 0 not in indices:
        raise Exception('Missing edgeLoop')
    result = [[loop[indices[i]], loop[indices[i] + 1:indices[i + 1]]] for i in range(len(indices) - 1)]
    if not sum([[a] + b for a, b in result], []) == loop:
        raise Exception('Error in _group_loop')
    return result