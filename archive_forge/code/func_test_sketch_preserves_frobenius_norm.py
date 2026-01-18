import numpy as np
from numpy.testing import assert_, assert_equal
from scipy.linalg import clarkson_woodruff_transform
from scipy.linalg._sketches import cwt_matrix
from scipy.sparse import issparse, rand
from scipy.sparse.linalg import norm
def test_sketch_preserves_frobenius_norm(self):
    n_errors = 0
    for A in self.test_matrices:
        if issparse(A):
            true_norm = norm(A)
        else:
            true_norm = np.linalg.norm(A)
        for seed in self.seeds:
            sketch = clarkson_woodruff_transform(A, self.n_sketch_rows, seed=seed)
            if issparse(sketch):
                sketch_norm = norm(sketch)
            else:
                sketch_norm = np.linalg.norm(sketch)
            if np.abs(true_norm - sketch_norm) > 0.1 * true_norm:
                n_errors += 1
    assert_(n_errors == 0)