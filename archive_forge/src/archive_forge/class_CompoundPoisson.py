import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
class CompoundPoisson:
    """nobs iid compound poisson distributions, not a process in time
    """

    def __init__(self, lambd, randfn=np.random.normal):
        if len(lambd) != len(randfn):
            raise ValueError('lambd and randfn need to have the same number of elements')
        self.nobj = len(lambd)
        self.randfn = randfn
        self.lambd = np.asarray(lambd)

    def simulate(self, nobs, nrepl=1):
        nobj = self.nobj
        x = np.zeros((nrepl, nobs, nobj))
        N = np.random.poisson(self.lambd[None, None, :], size=(nrepl, nobs, nobj))
        for io in range(nobj):
            randfnc = self.randfn[io]
            nc = N[:, :, io]
            rvs = randfnc(size=(nrepl, nobs, np.max(nc)))
            print('rvs.sum()', rvs.sum(), rvs.shape)
            xio = rvs.cumsum(-1)[np.arange(nrepl)[:, None], np.arange(nobs), nc - 1]
            x[:, :, io] = xio
        x[N == 0] = 0
        return (x, N)