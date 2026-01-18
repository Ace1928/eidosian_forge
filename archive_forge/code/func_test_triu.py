from numpy.testing import assert_equal
from scipy.sparse import csr_matrix, csr_array, sparray
import numpy as np
from scipy.sparse import _extract
def test_triu(self):
    for A in self.cases:
        B = A.toarray()
        for k in [-3, -2, -1, 0, 1, 2, 3]:
            assert_equal(_extract.triu(A, k=k).toarray(), np.triu(B, k=k))