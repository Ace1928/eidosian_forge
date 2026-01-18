import numpy as np
from numpy.testing import assert_, assert_equal
from scipy.linalg import clarkson_woodruff_transform
from scipy.linalg._sketches import cwt_matrix
from scipy.sparse import issparse, rand
from scipy.sparse.linalg import norm
def test_sketch_preserves_vector_norm(self):
    n_errors = 0
    n_sketch_rows = int(np.ceil(2.0 / (0.01 * 0.5 ** 2)))
    true_norm = np.linalg.norm(self.x)
    for seed in self.seeds:
        sketch = clarkson_woodruff_transform(self.x, n_sketch_rows, seed=seed)
        sketch_norm = np.linalg.norm(sketch)
        if np.abs(true_norm - sketch_norm) > 0.5 * true_norm:
            n_errors += 1
    assert_(n_errors == 0)