import numpy as np
def S_white(x, d):
    """simple white heteroscedasticity robust covariance
    note: calculating this way is very inefficient, just for cross-checking
    """
    r = np.ones(d.shape[1])
    return aggregate_cov(x, d, r=r, weights=None)