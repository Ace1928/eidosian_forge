import numpy as np
import matplotlib.pyplot as plt
class JumpDiffusionMerton:
    """

    Example
    -------
    mu=.00     # deterministic drift
    sig=.20 # Gaussian component
    l=3.45 # Poisson process arrival rate
    a=0 # drift of log-jump
    D=.2 # st.dev of log-jump

    X = JumpDiffusionMerton().simulate(mu,sig,lambd,a,D,ts,nrepl)

    plt.figure()
    plt.plot(X.T)
    plt.title('Merton jump-diffusion')


    """

    def __init__(self):
        pass

    def simulate(self, m, s, lambd, a, D, ts, nrepl):
        T = ts[-1]
        n_jumps = np.random.poisson(lambd * T, size=(nrepl, 1))
        jumps = []
        nobs = len(ts)
        jumps = np.zeros((nrepl, nobs))
        for j in range(nrepl):
            t = T * np.random.rand(n_jumps[j])
            t = np.sort(t, 0)
            S = a + D * np.random.randn(n_jumps[j], 1)
            CumS = np.cumsum(S)
            jumps_ts = np.zeros(nobs)
            for n in range(nobs):
                Events = np.sum(t <= ts[n]) - 1
                jumps_ts[n] = 0
                if Events > 0:
                    jumps_ts[n] = CumS[Events]
            jumps[j, :] = jumps_ts
        D_Diff = np.zeros((nrepl, nobs))
        for k in range(nobs):
            Dt = ts[k]
            if k > 1:
                Dt = ts[k] - ts[k - 1]
            D_Diff[:, k] = m * Dt + s * np.sqrt(Dt) * np.random.randn(nrepl)
        x = np.hstack((np.zeros((nrepl, 1)), np.cumsum(D_Diff, 1) + jumps))
        return x