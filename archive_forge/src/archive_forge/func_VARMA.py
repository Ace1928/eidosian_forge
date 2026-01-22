import numpy as np
from scipy import signal
def VARMA(x, B, C, const=0):
    """ multivariate linear filter

    x (TxK)
    B (PxKxK)

    xhat(t,i) = sum{_p}sum{_k} { x(t-P:t,:) .* B(:,:,i) } +
                sum{_q}sum{_k} { e(t-Q:t,:) .* C(:,:,i) }for all i = 0,K-1

    """
    P = B.shape[0]
    Q = C.shape[0]
    T = x.shape[0]
    xhat = np.zeros(x.shape)
    e = np.zeros(x.shape)
    start = max(P, Q)
    for t in range(start, T):
        xhat[t, :] = const + (x[t - P:t, :, np.newaxis] * B).sum(axis=1).sum(axis=0) + (e[t - Q:t, :, np.newaxis] * C).sum(axis=1).sum(axis=0)
        e[t, :] = x[t, :] - xhat[t, :]
    return (xhat, e)