import numpy as np
class ConstraintsPenalty:
    """
    Penalty applied to linear transformation of parameters

    Parameters
    ----------
    penalty: instance of penalty function
        currently this requires an instance of a univariate, vectorized
        penalty class
    weights : None or ndarray
        weights for adding penalties of transformed params
    restriction : None or ndarray
        If it is not None, then restriction defines a linear transformation
        of the parameters. The penalty function is applied to each transformed
        parameter independently.

    Notes
    -----
    `restrictions` allows us to impose penalization on contrasts or stochastic
    constraints of the original parameters.
    Examples for these contrast are difference penalities or all pairs
    penalties.
    """

    def __init__(self, penalty, weights=None, restriction=None):
        self.penalty = penalty
        if weights is None:
            self.weights = 1.0
        else:
            self.weights = weights
        if restriction is not None:
            restriction = np.asarray(restriction)
        self.restriction = restriction

    def func(self, params):
        """evaluate penalty function at params

        Parameter
        ---------
        params : ndarray
            array of parameters at which derivative is evaluated

        Returns
        -------
        deriv2 : ndarray
            value(s) of penalty function
        """
        if self.restriction is not None:
            params = self.restriction.dot(params)
        value = self.penalty.func(params)
        return (self.weights * value.T).T.sum(0)

    def deriv(self, params):
        """first derivative of penalty function w.r.t. params

        Parameter
        ---------
        params : ndarray
            array of parameters at which derivative is evaluated

        Returns
        -------
        deriv2 : ndarray
            array of first partial derivatives
        """
        if self.restriction is not None:
            params = self.restriction.dot(params)
        value = self.penalty.deriv(params)
        if self.restriction is not None:
            return self.weights * value.T.dot(self.restriction)
        else:
            return self.weights * value.T
    grad = deriv

    def deriv2(self, params):
        """second derivative of penalty function w.r.t. params

        Parameter
        ---------
        params : ndarray
            array of parameters at which derivative is evaluated

        Returns
        -------
        deriv2 : ndarray, 2-D
            second derivative matrix
        """
        if self.restriction is not None:
            params = self.restriction.dot(params)
        value = self.penalty.deriv2(params)
        if self.restriction is not None:
            v = self.restriction.T * value * self.weights
            value = v.dot(self.restriction)
        else:
            value = np.diag(self.weights * value)
        return value