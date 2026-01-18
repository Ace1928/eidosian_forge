import pickle
import re
import sys
from collections.abc import Iterable, Sized
from functools import partial
from io import StringIO
from itertools import chain, product
from types import GeneratorType
import numpy as np
import pytest
from scipy.stats import bernoulli, expon, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import (
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import (
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.neighbors import KernelDensity, KNeighborsClassifier, LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
def test_param_sampler():
    param_distributions = {'kernel': ['rbf', 'linear'], 'C': uniform(0, 1)}
    sampler = ParameterSampler(param_distributions=param_distributions, n_iter=10, random_state=0)
    samples = [x for x in sampler]
    assert len(samples) == 10
    for sample in samples:
        assert sample['kernel'] in ['rbf', 'linear']
        assert 0 <= sample['C'] <= 1
    param_distributions = {'C': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    sampler = ParameterSampler(param_distributions=param_distributions, n_iter=3, random_state=0)
    assert [x for x in sampler] == [x for x in sampler]
    param_distributions = {'C': uniform(0, 1)}
    sampler = ParameterSampler(param_distributions=param_distributions, n_iter=10, random_state=0)
    assert [x for x in sampler] == [x for x in sampler]