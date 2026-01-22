from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logit
from scipy.stats import gmean
from ..utils.extmath import softmax
class HalfLogitLink(BaseLink):
    """Half the logit link function g(x)=1/2 * logit(x).

    Used for the exponential loss.
    """
    interval_y_pred = Interval(0, 1, False, False)

    def link(self, y_pred, out=None):
        out = logit(y_pred, out=out)
        out *= 0.5
        return out

    def inverse(self, raw_prediction, out=None):
        return expit(2 * raw_prediction, out)