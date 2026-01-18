import pickle
import numpy as np
import pytest
from sklearn.neighbors._quad_tree import _QuadTree
from sklearn.utils import check_random_state
@pytest.mark.parametrize('n_dimensions', (2, 3))
@pytest.mark.parametrize('protocol', (0, 1, 2))
def test_quad_tree_pickle(n_dimensions, protocol):
    rng = check_random_state(0)
    X = rng.random_sample((10, n_dimensions))
    tree = _QuadTree(n_dimensions=n_dimensions, verbose=0)
    tree.build_tree(X)
    s = pickle.dumps(tree, protocol=protocol)
    bt2 = pickle.loads(s)
    for x in X:
        cell_x_tree = tree.get_cell(x)
        cell_x_bt2 = bt2.get_cell(x)
        assert cell_x_tree == cell_x_bt2