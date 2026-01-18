import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from sklearn.neighbors._kd_tree import KDTree, KDTree32, KDTree64
from sklearn.neighbors.tests.test_ball_tree import get_dataset_for_binary_tree
from sklearn.utils.parallel import Parallel, delayed
Make sure that KDTree queries work when joblib memmaps.

    Non-regression test for #21685 and #21228.