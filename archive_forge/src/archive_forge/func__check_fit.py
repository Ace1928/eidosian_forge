import functools
from inspect import signature
import numpy as np
import pytest
from sklearn.base import BaseEstimator, is_regressor
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.utils import all_estimators
from sklearn.utils._testing import set_random_state
from sklearn.utils.estimator_checks import (
from sklearn.utils.validation import check_is_fitted
def _check_fit(self):
    check_is_fitted(self)