import networkx as nx
from networkx.utils import np_random_state
@np_random_state(7)
def _fruchterman_reingold(A, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, dim=2, seed=None):
    import numpy as np
    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = 'fruchterman_reingold() takes an adjacency matrix as input'
        raise nx.NetworkXError(msg) from err
    if pos is None:
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        pos = pos.astype(A.dtype)
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    dt = t / (iterations + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
    for iteration in range(iterations):
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        np.clip(distance, 0.01, None, out=distance)
        displacement = np.einsum('ijk,ij->ik', delta, k * k / distance ** 2 - A * distance / k)
        length = np.linalg.norm(displacement, axis=-1)
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = np.einsum('ij,i->ij', displacement, t / length)
        if fixed is not None:
            delta_pos[fixed] = 0.0
        pos += delta_pos
        t -= dt
        if np.linalg.norm(delta_pos) / nnodes < threshold:
            break
    return pos