import unittest
import numpy as np
from scipy import sparse
from pygsp import graphs, utils
def test_symmetrize(self):
    W = sparse.random(100, 100, random_state=42)
    for method in ['average', 'maximum', 'fill', 'tril', 'triu']:
        W1 = utils.symmetrize(W, method=method)
        W2 = utils.symmetrize(W.toarray(), method=method)
        np.testing.assert_equal(W1.toarray(), W2)
    self.assertRaises(ValueError, utils.symmetrize, W, 'sum')