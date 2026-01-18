import pickle
import numpy as np
import pytest
from sklearn.neighbors._quad_tree import _QuadTree
from sklearn.utils import check_random_state
@pytest.mark.parametrize('n_dimensions', (2, 3))
def test_qt_insert_duplicate(n_dimensions):
    rng = check_random_state(0)
    X = rng.random_sample((10, n_dimensions))
    Xd = np.r_[X, X[:5]]
    tree = _QuadTree(n_dimensions=n_dimensions, verbose=0)
    tree.build_tree(Xd)
    cumulative_size = tree.cumulative_size
    leafs = tree.leafs
    for i, x in enumerate(X):
        cell_id = tree.get_cell(x)
        assert leafs[cell_id]
        assert cumulative_size[cell_id] == 1 + (i < 5)