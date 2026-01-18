import warnings
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets, linear_model
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._least_angle import _lars_path_residues
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
def test_rank_deficient_design():
    y = [5, 0, 5]
    for X in ([[5, 0], [0, 5], [10, 10]], [[10, 10, 0], [1e-32, 0, 0], [0, 0, 1]]):
        lars = linear_model.LassoLars(0.1)
        coef_lars_ = lars.fit(X, y).coef_
        obj_lars = 1.0 / (2.0 * 3.0) * linalg.norm(y - np.dot(X, coef_lars_)) ** 2 + 0.1 * linalg.norm(coef_lars_, 1)
        coord_descent = linear_model.Lasso(0.1, tol=1e-06)
        coef_cd_ = coord_descent.fit(X, y).coef_
        obj_cd = 1.0 / (2.0 * 3.0) * linalg.norm(y - np.dot(X, coef_cd_)) ** 2 + 0.1 * linalg.norm(coef_cd_, 1)
        assert obj_lars < obj_cd * (1.0 + 1e-08)