import numpy as np
from pygsp import utils
def _frame_matrix(self, g, normalize=False):
    """Create the GWFT frame.

        Parameters
        ----------
        g : window

        Returns
        -------
        F : ndarray
            Frame
        """
    N = self.N
    U = self.U
    if self.N > 256:
        logger.warning('It will create a big matrix. You can use other methods.')
    ghat = np.dot(U.T, g)
    Ftrans = np.sqrt(N) * np.dot(U, np.kron(np.ones(N), ghat) * U.T)
    F = utils.repmatline(Ftrans, 1, N) * np.kron(np.ones(N), np.kron(np.ones(N), 1.0 / U[:, 0]))
    if normalize:
        F /= np.kron((np.ones(N), np.sqrt(np.sum(np.power(np.abs(F), 2), axis=0))))
    return F