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
def test_warm_start_effectiveness():
    nca_warm = NeighborhoodComponentsAnalysis(warm_start=True, random_state=0)
    nca_warm.fit(iris_data, iris_target)
    transformation_warm = nca_warm.components_
    nca_warm.max_iter = 1
    nca_warm.fit(iris_data, iris_target)
    transformation_warm_plus_one = nca_warm.components_
    nca_cold = NeighborhoodComponentsAnalysis(warm_start=False, random_state=0)
    nca_cold.fit(iris_data, iris_target)
    transformation_cold = nca_cold.components_
    nca_cold.max_iter = 1
    nca_cold.fit(iris_data, iris_target)
    transformation_cold_plus_one = nca_cold.components_
    diff_warm = np.sum(np.abs(transformation_warm_plus_one - transformation_warm))
    diff_cold = np.sum(np.abs(transformation_cold_plus_one - transformation_cold))
    assert diff_warm < 3.0, 'Transformer changed significantly after one iteration even though it was warm-started.'
    assert diff_cold > diff_warm, 'Cold-started transformer changed less significantly than warm-started transformer after one iteration.'