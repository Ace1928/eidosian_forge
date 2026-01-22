import networkx as nx
from networkx.utils import np_random_state
@np_random_state(7)
def _sparse_fruchterman_reingold(A, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, dim=2, seed=None):
    import numpy as np
    import scipy as sp
    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = 'fruchterman_reingold() takes an adjacency matrix as input'
        raise nx.NetworkXError(msg) from err
    try:
        A = A.tolil()
    except AttributeError:
        A = sp.sparse.coo_array(A).tolil()
    if pos is None:
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        pos = pos.astype(A.dtype)
    if fixed is None:
        fixed = []
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    dt = t / (iterations + 1)
    displacement = np.zeros((dim, nnodes))
    for iteration in range(iterations):
        displacement *= 0
        for i in range(A.shape[0]):
            if i in fixed:
                continue
            delta = (pos[i] - pos).T
            distance = np.sqrt((delta ** 2).sum(axis=0))
            distance = np.where(distance < 0.01, 0.01, distance)
            Ai = A.getrowview(i).toarray()
            displacement[:, i] += (delta * (k * k / distance ** 2 - Ai * distance / k)).sum(axis=1)
        length = np.sqrt((displacement ** 2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = (displacement * t / length).T
        pos += delta_pos
        t -= dt
        if np.linalg.norm(delta_pos) / nnodes < threshold:
            break
    return pos