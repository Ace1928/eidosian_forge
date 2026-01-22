from collections import defaultdict
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.tsatools import duplication_matrix, lagmat, vec
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import (
def _endog_matrices(endog, exog, exog_coint, diff_lags, deterministic, seasons=0, first_season=0):
    """
    Returns different matrices needed for parameter estimation.

    Compare p. 186 in [1]_. The returned matrices consist of elements of the
    data as well as elements representing deterministic terms. A tuple of
    consisting of these matrices is returned.

    Parameters
    ----------
    endog : ndarray (neqs x nobs_tot)
        The whole sample including the presample.
    exog : ndarray (nobs_tot x neqs) or None
        Deterministic terms outside the cointegration relation.
    exog_coint : ndarray (nobs_tot x neqs) or None
        Deterministic terms inside the cointegration relation.
    diff_lags : int
        Number of lags in the VEC representation.
    deterministic : str {``"n"``, ``"co"``, ``"ci"``, ``"lo"``, ``"li"``}
        * ``"n"`` - no deterministic terms
        * ``"co"`` - constant outside the cointegration relation
        * ``"ci"`` - constant within the cointegration relation
        * ``"lo"`` - linear trend outside the cointegration relation
        * ``"li"`` - linear trend within the cointegration relation

        Combinations of these are possible (e.g. ``"cili"`` or ``"colo"`` for
        linear trend with intercept). See the docstring of the
        :class:`VECM`-class for more information.
    seasons : int, default: 0
        Number of periods in a seasonal cycle. 0 (default) means no seasons.
    first_season : int, default: 0
        The season of the first observation. `0` means first season, `1` means
        second season, ..., `seasons-1` means the last season.

    Returns
    -------
    y_1_T : ndarray (neqs x nobs)
        The (transposed) data without the presample.
        `.. math:: (y_1, \\ldots, y_T)
    delta_y_1_T : ndarray (neqs x nobs)
        The first differences of endog.
        `.. math:: (y_1, \\ldots, y_T) - (y_0, \\ldots, y_{T-1})
    y_lag1 : ndarray (neqs x nobs)
        (dimensions assuming no deterministic terms are given)
        Endog of the previous period (lag 1).
        `.. math:: (y_0, \\ldots, y_{T-1})
    delta_x : ndarray (k_ar_diff*neqs x nobs)
        (dimensions assuming no deterministic terms are given)
        Lagged differenced endog, used as regressor for the short term
        equation.

    References
    ----------
    .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
    """
    deterministic = string_like(deterministic, 'deterministic')
    p = diff_lags + 1
    y = endog
    K = y.shape[0]
    y_1_T = y[:, p:]
    T = y_1_T.shape[1]
    delta_y = np.diff(y)
    delta_y_1_T = delta_y[:, p - 1:]
    y_lag1 = y[:, p - 1:-1]
    if 'co' in deterministic and 'ci' in deterministic:
        raise ValueError("Both 'co' and 'ci' as deterministic terms given. " + 'Please choose one of the two.')
    y_lag1_stack = [y_lag1]
    if 'ci' in deterministic:
        y_lag1_stack.append(np.ones(T))
    if 'li' in deterministic:
        y_lag1_stack.append(_linear_trend(T, p, coint=True))
    if exog_coint is not None:
        y_lag1_stack.append(exog_coint[-T - 1:-1].T)
    y_lag1 = np.vstack(y_lag1_stack)
    delta_x = np.zeros((diff_lags * K, T))
    if diff_lags > 0:
        for j in range(delta_x.shape[1]):
            delta_x[:, j] = delta_y[:, j + p - 2:None if j - 1 < 0 else j - 1:-1].T.reshape(K * (p - 1))
    delta_x_stack = [delta_x]
    if 'co' in deterministic:
        delta_x_stack.append(np.ones(T))
    if seasons > 0:
        delta_x_stack.append(seasonal_dummies(seasons, delta_x.shape[1], first_period=first_season + diff_lags + 1, centered=True).T)
    if 'lo' in deterministic:
        delta_x_stack.append(_linear_trend(T, p))
    if exog is not None:
        delta_x_stack.append(exog[-T:].T)
    delta_x = np.vstack(delta_x_stack)
    return (y_1_T, delta_y_1_T, y_lag1, delta_x)