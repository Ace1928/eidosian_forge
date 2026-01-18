import numpy as np
from numpy.testing import assert_, assert_equal
from scipy.linalg import clarkson_woodruff_transform
from scipy.linalg._sketches import cwt_matrix
from scipy.sparse import issparse, rand
from scipy.sparse.linalg import norm
def test_seed_returns_identically(self):
    for A in self.test_matrices:
        for seed in self.seeds:
            sketch1 = clarkson_woodruff_transform(A, self.n_sketch_rows, seed=seed)
            sketch2 = clarkson_woodruff_transform(A, self.n_sketch_rows, seed=seed)
            if issparse(sketch1):
                sketch1 = sketch1.toarray()
            if issparse(sketch2):
                sketch2 = sketch2.toarray()
            assert_equal(sketch1, sketch2)