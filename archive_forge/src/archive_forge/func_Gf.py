import numpy as np
def Gf(T, ff):
    """
    Subroutine for the gradient of f using numerical derivatives.
    """
    k = T.shape[0]
    ep = 0.0001
    G = np.zeros((k, k))
    for r in range(k):
        for s in range(k):
            dT = np.zeros((k, k))
            dT[r, s] = ep
            G[r, s] = (ff(T + dT) - ff(T - dT)) / (2 * ep)
    return G