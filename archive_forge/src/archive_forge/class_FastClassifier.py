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
class FastClassifier(DummyClassifier):
    """Dummy classifier that accepts parameters a, b, ... z.

    These parameter don't affect the predictions and are useful for fast
    grid searching."""
    _parameter_constraints: dict = {**DummyClassifier._parameter_constraints, **{chr(key): 'no_validation' for key in range(ord('a'), ord('z') + 1)}}

    def __init__(self, strategy='stratified', random_state=None, constant=None, **kwargs):
        super().__init__(strategy=strategy, random_state=random_state, constant=constant)

    def get_params(self, deep=False):
        params = super().get_params(deep=deep)
        for char in range(ord('a'), ord('z') + 1):
            params[chr(char)] = 'whatever'
        return params