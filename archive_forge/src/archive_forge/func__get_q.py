import numpy as np
def _get_q(self, z):
    nobs = len(z)
    mask_neg = z < 0
    qq = np.empty(nobs)
    qq[mask_neg] = 1 - self.q
    qq[~mask_neg] = self.q
    return qq