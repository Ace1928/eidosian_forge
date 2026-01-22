import numpy as np
from statsmodels.robust import mad
from scipy.optimize import minimize_scalar
def _est_lambda(self, x, bounds=(-1, 2), method='guerrero', **kwargs):
    """
        Computes an estimate for the lambda parameter in the Box-Cox
        transformation using method.

        Parameters
        ----------
        x : array_like
            The untransformed data.
        bounds : tuple
            Numeric 2-tuple, that indicate the solution space for the lambda
            parameter. Default (-1, 2).
        method : {'guerrero', 'loglik'}
            The method by which to estimate lambda. Defaults to 'guerrero', but
            the profile likelihood ('loglik') is also available.
        **kwargs
            Options for the specified method.
            * For 'guerrero': window_length (int), the seasonality/grouping
              parameter. Scale ({'mad', 'sd'}), the dispersion measure. Options
              (dict), to be passed to the optimizer.
            * For 'loglik': Options (dict), to be passed to the optimizer.

        Returns
        -------
        lmbda : float
            The lambda parameter.
        """
    method = method.lower()
    if len(bounds) != 2:
        raise ValueError('Bounds of length {} not understood.'.format(len(bounds)))
    elif bounds[0] >= bounds[1]:
        raise ValueError('Lower bound exceeds upper bound.')
    if method == 'guerrero':
        lmbda = self._guerrero_cv(x, bounds=bounds, **kwargs)
    elif method == 'loglik':
        lmbda = self._loglik_boxcox(x, bounds=bounds, **kwargs)
    else:
        raise ValueError(f"Method '{method}' not understood.")
    return lmbda