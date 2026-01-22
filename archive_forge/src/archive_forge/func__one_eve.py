import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import kron, eye, dia_array
def _one_eve(self, k):
    """Return 1 eigenvector in Nd with multi-index `j`
        as a tensor product of the corresponding 1d eigenvectors. 
        """
    phi = [self._ev1d(j, n) for j, n in zip(k, self.grid_shape)]
    result = phi[0]
    for phi in phi[1:]:
        result = np.tensordot(result, phi, axes=0)
    return np.asarray(result).ravel()