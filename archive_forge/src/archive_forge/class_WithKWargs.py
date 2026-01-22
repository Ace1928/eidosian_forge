import re
from pprint import PrettyPrinter
import numpy as np
from sklearn.utils._pprint import _EstimatorPrettyPrinter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import config_context
class WithKWargs(BaseEstimator):

    def __init__(self, a='willchange', b='unchanged', **kwargs):
        self.a = a
        self.b = b
        self._other_params = {}
        self.set_params(**kwargs)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(self._other_params)
        return params

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
            self._other_params[key] = value
        return self