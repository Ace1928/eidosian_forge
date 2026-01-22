import warnings
from Bio import BiopythonDeprecationWarning
def _baum_welch_one(N, M, outputs, lp_initial, lp_transition, lp_emission, lpseudo_initial, lpseudo_transition, lpseudo_emission):
    """Execute one step for Baum-Welch algorithm (PRIVATE).

    Do one iteration of Baum-Welch based on a sequence of output.
    Changes the value for lp_initial, lp_transition and lp_emission in place.
    """
    T = len(outputs)
    fmat = _forward(N, T, lp_initial, lp_transition, lp_emission, outputs)
    bmat = _backward(N, T, lp_transition, lp_emission, outputs)
    lp_arc = np.zeros((N, N, T))
    for t in range(T):
        k = outputs[t]
        lp_traverse = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                lp = fmat[i][t] + lp_transition[i][j] + lp_emission[i][k] + bmat[j][t + 1]
                lp_traverse[i][j] = lp
        lp_arc[:, :, t] = lp_traverse - _logsum(lp_traverse)
    lp_arcout_t = np.zeros((N, T))
    for t in range(T):
        for i in range(N):
            lp_arcout_t[i][t] = _logsum(lp_arc[i, :, t])
    lp_arcout = np.zeros(N)
    for i in range(N):
        lp_arcout[i] = _logsum(lp_arcout_t[i, :])
    lp_initial = lp_arcout_t[:, 0]
    if lpseudo_initial is not None:
        lp_initial = _logvecadd(lp_initial, lpseudo_initial)
        lp_initial = lp_initial - _logsum(lp_initial)
    for i in range(N):
        for j in range(N):
            lp_transition[i][j] = _logsum(lp_arc[i, j, :]) - lp_arcout[i]
        if lpseudo_transition is not None:
            lp_transition[i] = _logvecadd(lp_transition[i], lpseudo_transition)
            lp_transition[i] = lp_transition[i] - _logsum(lp_transition[i])
    for i in range(N):
        ksum = np.zeros(M) + LOG0
        for t in range(T):
            k = outputs[t]
            for j in range(N):
                ksum[k] = logaddexp(ksum[k], lp_arc[i, j, t])
        ksum = ksum - _logsum(ksum)
        if lpseudo_emission is not None:
            ksum = _logvecadd(ksum, lpseudo_emission[i])
            ksum = ksum - _logsum(ksum)
        lp_emission[i, :] = ksum
    return _logsum(fmat[:, T])