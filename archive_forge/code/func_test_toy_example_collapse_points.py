import re
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.optimize import check_grad
from sklearn import clone
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
def test_toy_example_collapse_points():
    """Test on a toy example of three points that should collapse

    We build a simple example: two points from the same class and a point from
    a different class in the middle of them. On this simple example, the new
    (transformed) points should all collapse into one single point. Indeed, the
    objective is 2/(1 + exp(d/2)), with d the euclidean distance between the
    two samples from the same class. This is maximized for d=0 (because d>=0),
    with an objective equal to 1 (loss=-1.).

    """
    rng = np.random.RandomState(42)
    input_dim = 5
    two_points = rng.randn(2, input_dim)
    X = np.vstack([two_points, two_points.mean(axis=0)[np.newaxis, :]])
    y = [0, 0, 1]

    class LossStorer:

        def __init__(self, X, y):
            self.loss = np.inf
            self.fake_nca = NeighborhoodComponentsAnalysis()
            self.fake_nca.n_iter_ = np.inf
            self.X, y = self.fake_nca._validate_data(X, y, ensure_min_samples=2)
            y = LabelEncoder().fit_transform(y)
            self.same_class_mask = y[:, np.newaxis] == y[np.newaxis, :]

        def callback(self, transformation, n_iter):
            """Stores the last value of the loss function"""
            self.loss, _ = self.fake_nca._loss_grad_lbfgs(transformation, self.X, self.same_class_mask, -1.0)
    loss_storer = LossStorer(X, y)
    nca = NeighborhoodComponentsAnalysis(random_state=42, callback=loss_storer.callback)
    X_t = nca.fit_transform(X, y)
    print(X_t)
    assert_array_almost_equal(X_t - X_t[0], 0.0)
    assert abs(loss_storer.loss + 1) < 1e-10