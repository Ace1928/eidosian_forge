import numpy as np
from scipy.sparse.linalg import aslinearoperator
def _algorithm_2_2(A, AT, t):
    """
    This is Algorithm 2.2.

    Parameters
    ----------
    A : ndarray or other linear operator
        A linear operator that can produce matrix products.
    AT : ndarray or other linear operator
        The transpose of A.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.

    Returns
    -------
    g : sequence
        A non-negative decreasing vector
        such that g[j] is a lower bound for the 1-norm
        of the column of A of jth largest 1-norm.
        The first entry of this vector is therefore a lower bound
        on the 1-norm of the linear operator A.
        This sequence has length t.
    ind : sequence
        The ith entry of ind is the index of the column A whose 1-norm
        is given by g[i].
        This sequence of indices has length t, and its entries are
        chosen from range(n), possibly with repetition,
        where n is the order of the operator A.

    Notes
    -----
    This algorithm is mainly for testing.
    It uses the 'ind' array in a way that is similar to
    its usage in algorithm 2.4. This algorithm 2.2 may be easier to test,
    so it gives a chance of uncovering bugs related to indexing
    which could have propagated less noticeably to algorithm 2.4.

    """
    A_linear_operator = aslinearoperator(A)
    AT_linear_operator = aslinearoperator(AT)
    n = A_linear_operator.shape[0]
    X = np.ones((n, t))
    if t > 1:
        X[:, 1:] = np.random.randint(0, 2, size=(n, t - 1)) * 2 - 1
    X /= float(n)
    g_prev = None
    h_prev = None
    k = 1
    ind = range(t)
    while True:
        Y = np.asarray(A_linear_operator.matmat(X))
        g = _sum_abs_axis0(Y)
        best_j = np.argmax(g)
        g.sort()
        g = g[::-1]
        S = sign_round_up(Y)
        Z = np.asarray(AT_linear_operator.matmat(S))
        h = _max_abs_axis1(Z)
        if k >= 2:
            if less_than_or_close(max(h), np.dot(Z[:, best_j], X[:, best_j])):
                break
        ind = np.argsort(h)[::-1][:t]
        h = h[ind]
        for j in range(t):
            X[:, j] = elementary_vector(n, ind[j])
        if k >= 2:
            if not less_than_or_close(g_prev[0], h_prev[0]):
                raise Exception('invariant (2.2) is violated')
            if not less_than_or_close(h_prev[0], g[0]):
                raise Exception('invariant (2.2) is violated')
        if k >= 3:
            for j in range(t):
                if not less_than_or_close(g[j], g_prev[j]):
                    raise Exception('invariant (2.3) is violated')
        g_prev = g
        h_prev = h
        k += 1
    return (g, ind)