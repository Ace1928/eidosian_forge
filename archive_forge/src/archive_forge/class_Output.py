import os
import numpy
from warnings import warn
from scipy.odr import __odrpack
class Output:
    """
    The Output class stores the output of an ODR run.

    Attributes
    ----------
    beta : ndarray
        Estimated parameter values, of shape (q,).
    sd_beta : ndarray
        Standard deviations of the estimated parameters, of shape (p,).
    cov_beta : ndarray
        Covariance matrix of the estimated parameters, of shape (p,p).
        Note that this `cov_beta` is not scaled by the residual variance 
        `res_var`, whereas `sd_beta` is. This means 
        ``np.sqrt(np.diag(output.cov_beta * output.res_var))`` is the same 
        result as `output.sd_beta`.
    delta : ndarray, optional
        Array of estimated errors in input variables, of same shape as `x`.
    eps : ndarray, optional
        Array of estimated errors in response variables, of same shape as `y`.
    xplus : ndarray, optional
        Array of ``x + delta``.
    y : ndarray, optional
        Array ``y = fcn(x + delta)``.
    res_var : float, optional
        Residual variance.
    sum_square : float, optional
        Sum of squares error.
    sum_square_delta : float, optional
        Sum of squares of delta error.
    sum_square_eps : float, optional
        Sum of squares of eps error.
    inv_condnum : float, optional
        Inverse condition number (cf. ODRPACK UG p. 77).
    rel_error : float, optional
        Relative error in function values computed within fcn.
    work : ndarray, optional
        Final work array.
    work_ind : dict, optional
        Indices into work for drawing out values (cf. ODRPACK UG p. 83).
    info : int, optional
        Reason for returning, as output by ODRPACK (cf. ODRPACK UG p. 38).
    stopreason : list of str, optional
        `info` interpreted into English.

    Notes
    -----
    Takes one argument for initialization, the return value from the
    function `~scipy.odr.odr`. The attributes listed as "optional" above are
    only present if `~scipy.odr.odr` was run with ``full_output=1``.

    """

    def __init__(self, output):
        self.beta = output[0]
        self.sd_beta = output[1]
        self.cov_beta = output[2]
        if len(output) == 4:
            self.__dict__.update(output[3])
            self.stopreason = _report_error(self.info)

    def pprint(self):
        """ Pretty-print important results.
        """
        print('Beta:', self.beta)
        print('Beta Std Error:', self.sd_beta)
        print('Beta Covariance:', self.cov_beta)
        if hasattr(self, 'info'):
            print('Residual Variance:', self.res_var)
            print('Inverse Condition #:', self.inv_condnum)
            print('Reason(s) for Halting:')
            for r in self.stopreason:
                print('  %s' % r)