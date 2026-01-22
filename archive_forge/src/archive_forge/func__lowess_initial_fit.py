import numpy as np
from numpy.linalg import lstsq
def _lowess_initial_fit(x_copy, y_copy, k, n):
    """
    The initial weighted local linear regression for lowess.

    Parameters
    ----------
    x_copy : 1-d ndarray
        The x-values/exogenous part of the data being smoothed
    y_copy : 1-d ndarray
        The y-values/ endogenous part of the data being smoothed
   k : int
        The number of data points which affect the linear fit for
        each estimated point
    n : int
        The total number of points

    Returns
    -------
    fitted : 1-d ndarray
        The fitted y-values
    weights : 2-d ndarray
        An n by k array. The contribution to the weights in the
        local linear fit coming from the distances between the
        x-values

   """
    weights = np.zeros((n, k), dtype=x_copy.dtype)
    nn_indices = [0, k]
    X = np.ones((k, 2))
    fitted = np.zeros(n)
    for i in range(n):
        left_width = x_copy[i] - x_copy[nn_indices[0]]
        right_width = x_copy[nn_indices[1] - 1] - x_copy[i]
        width = max(left_width, right_width)
        _lowess_wt_standardize(weights[i, :], x_copy[nn_indices[0]:nn_indices[1]], x_copy[i], width)
        _lowess_tricube(weights[i, :])
        weights[i, :] = np.sqrt(weights[i, :])
        X[:, 1] = x_copy[nn_indices[0]:nn_indices[1]]
        y_i = weights[i, :] * y_copy[nn_indices[0]:nn_indices[1]]
        beta = lstsq(weights[i, :].reshape(k, 1) * X, y_i, rcond=-1)[0]
        fitted[i] = beta[0] + beta[1] * x_copy[i]
        _lowess_update_nn(x_copy, nn_indices, i + 1)
    return (fitted, weights)