from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logit
from scipy.stats import gmean
from ..utils.extmath import softmax
class LogLink(BaseLink):
    """The log link function g(x)=log(x)."""
    interval_y_pred = Interval(0, np.inf, False, False)

    def link(self, y_pred, out=None):
        return np.log(y_pred, out=out)

    def inverse(self, raw_prediction, out=None):
        return np.exp(raw_prediction, out=out)