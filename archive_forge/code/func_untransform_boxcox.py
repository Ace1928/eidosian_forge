import numpy as np
from statsmodels.robust import mad
from scipy.optimize import minimize_scalar
def untransform_boxcox(self, x, lmbda, method='naive'):
    """
        Back-transforms the Box-Cox transformed data array, by means of the
        indicated method. The provided argument lmbda should be the lambda
        parameter that was used to initially transform the data.

        Parameters
        ----------
        x : array_like
            The transformed series.
        lmbda : float
            The lambda parameter that was used to transform the series.
        method : {'naive'}
            Indicates the method to be used in the untransformation. Defaults
            to 'naive', which reverses the transformation.

            NOTE: 'naive' is implemented natively, while other methods may be
            available in subclasses!

        Returns
        -------
        y : array_like
            The untransformed series.
        """
    method = method.lower()
    x = np.asarray(x)
    if method == 'naive':
        if np.isclose(lmbda, 0.0):
            y = np.exp(x)
        else:
            y = np.power(lmbda * x + 1, 1.0 / lmbda)
    else:
        raise ValueError(f"Method '{method}' not understood.")
    return y