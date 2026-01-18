import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import kron, eye, dia_array
def tosparse(self):
    from scipy.sparse import diags
    return diags([self._diag1, self._diag0, self._diag1], [-1, 0, 1], shape=self.shape, dtype=self.dtype)