import pickle
import numpy as np
import pytest
from sklearn.neighbors._quad_tree import _QuadTree
from sklearn.utils import check_random_state
def test_quadtree_boundary_computation():
    Xs = []
    Xs.append(np.array([[-1, 1], [-4, -1]], dtype=np.float32))
    Xs.append(np.array([[0, 0], [0, 0]], dtype=np.float32))
    Xs.append(np.array([[-1, -2], [-4, 0]], dtype=np.float32))
    Xs.append(np.array([[-1e-06, 1e-06], [-4e-06, -1e-06]], dtype=np.float32))
    for X in Xs:
        tree = _QuadTree(n_dimensions=2, verbose=0)
        tree.build_tree(X)
        tree._check_coherence()