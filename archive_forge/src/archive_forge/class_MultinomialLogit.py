from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logit
from scipy.stats import gmean
from ..utils.extmath import softmax
class MultinomialLogit(BaseLink):
    """The symmetric multinomial logit function.

    Convention:
        - y_pred.shape = raw_prediction.shape = (n_samples, n_classes)

    Notes:
        - The inverse link h is the softmax function.
        - The sum is over the second axis, i.e. axis=1 (n_classes).

    We have to choose additional constraints in order to make

        y_pred[k] = exp(raw_pred[k]) / sum(exp(raw_pred[k]), k=0..n_classes-1)

    for n_classes classes identifiable and invertible.
    We choose the symmetric side constraint where the geometric mean response
    is set as reference category, see [2]:

    The symmetric multinomial logit link function for a single data point is
    then defined as

        raw_prediction[k] = g(y_pred[k]) = log(y_pred[k]/gmean(y_pred))
        = log(y_pred[k]) - mean(log(y_pred)).

    Note that this is equivalent to the definition in [1] and implies mean
    centered raw predictions:

        sum(raw_prediction[k], k=0..n_classes-1) = 0.

    For linear models with raw_prediction = X @ coef, this corresponds to
    sum(coef[k], k=0..n_classes-1) = 0, i.e. the sum over classes for every
    feature is zero.

    Reference
    ---------
    .. [1] Friedman, Jerome; Hastie, Trevor; Tibshirani, Robert. "Additive
        logistic regression: a statistical view of boosting" Ann. Statist.
        28 (2000), no. 2, 337--407. doi:10.1214/aos/1016218223.
        https://projecteuclid.org/euclid.aos/1016218223

    .. [2] Zahid, Faisal Maqbool and Gerhard Tutz. "Ridge estimation for
        multinomial logit models with symmetric side constraints."
        Computational Statistics 28 (2013): 1017-1034.
        http://epub.ub.uni-muenchen.de/11001/1/tr067.pdf
    """
    is_multiclass = True
    interval_y_pred = Interval(0, 1, False, False)

    def symmetrize_raw_prediction(self, raw_prediction):
        return raw_prediction - np.mean(raw_prediction, axis=1)[:, np.newaxis]

    def link(self, y_pred, out=None):
        gm = gmean(y_pred, axis=1)
        return np.log(y_pred / gm[:, np.newaxis], out=out)

    def inverse(self, raw_prediction, out=None):
        if out is None:
            return softmax(raw_prediction, copy=True)
        else:
            np.copyto(out, raw_prediction)
            softmax(out, copy=False)
            return out