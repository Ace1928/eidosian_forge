from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logit
from scipy.stats import gmean
from ..utils.extmath import softmax
class LogitLink(BaseLink):
    """The logit link function g(x)=logit(x)."""
    interval_y_pred = Interval(0, 1, False, False)

    def link(self, y_pred, out=None):
        return logit(y_pred, out=out)

    def inverse(self, raw_prediction, out=None):
        return expit(raw_prediction, out=out)