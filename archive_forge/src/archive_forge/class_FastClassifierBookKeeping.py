from math import ceil
import numpy as np
import pytest
from scipy.stats import expon, norm, randint
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
from sklearn.model_selection._search_successive_halving import (
from sklearn.model_selection.tests.test_search import (
from sklearn.svm import SVC, LinearSVC
class FastClassifierBookKeeping(FastClassifier):

    def fit(self, X, y):
        passed_n_samples_fit.append(X.shape[0])
        return super().fit(X, y)

    def predict(self, X):
        passed_n_samples_predict.append(X.shape[0])
        return super().predict(X)

    def set_params(self, **params):
        passed_params.append(params)
        return super().set_params(**params)