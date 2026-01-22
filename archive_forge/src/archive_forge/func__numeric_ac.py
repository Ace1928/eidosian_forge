import networkx as nx
from networkx.algorithms.assortativity.mixing import (
from networkx.algorithms.assortativity.pairs import node_degree_xy
def _numeric_ac(M, mapping):
    import numpy as np
    if M.sum() != 1.0:
        M = M / M.sum()
    x = np.array(list(mapping.keys()))
    y = x
    idx = list(mapping.values())
    a = M.sum(axis=0)
    b = M.sum(axis=1)
    vara = (a[idx] * x ** 2).sum() - (a[idx] * x).sum() ** 2
    varb = (b[idx] * y ** 2).sum() - (b[idx] * y).sum() ** 2
    xy = np.outer(x, y)
    ab = np.outer(a[idx], b[idx])
    return (xy * (M - ab)).sum() / np.sqrt(vara * varb)