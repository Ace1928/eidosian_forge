from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logit
from scipy.stats import gmean
from ..utils.extmath import softmax
class BaseLink(ABC):
    """Abstract base class for differentiable, invertible link functions.

    Convention:
        - link function g: raw_prediction = g(y_pred)
        - inverse link h: y_pred = h(raw_prediction)

    For (generalized) linear models, `raw_prediction = X @ coef` is the so
    called linear predictor, and `y_pred = h(raw_prediction)` is the predicted
    conditional (on X) expected value of the target `y_true`.

    The methods are not implemented as staticmethods in case a link function needs
    parameters.
    """
    is_multiclass = False
    interval_y_pred = Interval(-np.inf, np.inf, False, False)

    @abstractmethod
    def link(self, y_pred, out=None):
        """Compute the link function g(y_pred).

        The link function maps (predicted) target values to raw predictions,
        i.e. `g(y_pred) = raw_prediction`.

        Parameters
        ----------
        y_pred : array
            Predicted target values.
        out : array
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or None,
            a freshly-allocated array is returned.

        Returns
        -------
        out : array
            Output array, element-wise link function.
        """

    @abstractmethod
    def inverse(self, raw_prediction, out=None):
        """Compute the inverse link function h(raw_prediction).

        The inverse link function maps raw predictions to predicted target
        values, i.e. `h(raw_prediction) = y_pred`.

        Parameters
        ----------
        raw_prediction : array
            Raw prediction values (in link space).
        out : array
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or None,
            a freshly-allocated array is returned.

        Returns
        -------
        out : array
            Output array, element-wise inverse link function.
        """