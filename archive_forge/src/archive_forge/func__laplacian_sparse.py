import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
def _laplacian_sparse(graph, normed, axis, copy, form, dtype, symmetrized):
    del form
    if dtype is None:
        dtype = graph.dtype
    needs_copy = False
    if graph.format in ('lil', 'dok'):
        m = graph.tocoo()
    else:
        m = graph
        if copy:
            needs_copy = True
    if symmetrized:
        m += m.T.conj()
    w = np.asarray(m.sum(axis=axis)).ravel() - m.diagonal()
    if normed:
        m = m.tocoo(copy=needs_copy)
        isolated_node_mask = w == 0
        w = np.where(isolated_node_mask, 1, np.sqrt(w))
        m.data /= w[m.row]
        m.data /= w[m.col]
        m.data *= -1
        m.setdiag(1 - isolated_node_mask)
    else:
        if m.format == 'dia':
            m = m.copy()
        else:
            m = m.tocoo(copy=needs_copy)
        m.data *= -1
        m.setdiag(w)
    return (m.astype(dtype, copy=False), w.astype(dtype))