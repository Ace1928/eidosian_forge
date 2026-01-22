import numpy as np
def __calc(self):
    A = self.A
    M = A - np.mean(A, axis=0)
    N = M / np.std(M, axis=0)
    self.M = M
    self.N = N
    self._eig = None