import networkx as nx
from networkx.utils import np_random_state
def _sparse_spectral(A, dim=2):
    import numpy as np
    import scipy as sp
    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = 'sparse_spectral() takes an adjacency matrix as input'
        raise nx.NetworkXError(msg) from err
    D = sp.sparse.csr_array(sp.sparse.spdiags(A.sum(axis=1), 0, nnodes, nnodes))
    L = D - A
    k = dim + 1
    ncv = max(2 * k + 1, int(np.sqrt(nnodes)))
    eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(L, k, which='SM', ncv=ncv)
    index = np.argsort(eigenvalues)[1:k]
    return np.real(eigenvectors[:, index])