import numpy as np
from scipy import signal
from statsmodels.tsa.tsatools import lagmat
def vstackarma_minus1(self):
    """stack ar and lagpolynomial vertically in 2d array

        """
    a = np.concatenate((self.ar[1:], self.ma[1:]), 0)
    return a.reshape(-1, self.nvarall)