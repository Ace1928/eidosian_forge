from statsmodels.regression.linear_model import OLS
import numpy as np
class RegularizedInvCovariance:
    """
    Class for estimating regularized inverse covariance with
    nodewise regression

    Parameters
    ----------
    exog : array_like
        A weighted design matrix for covariance

    Attributes
    ----------
    exog : array_like
        A weighted design matrix for covariance
    alpha : scalar
        Regularizing constant
    """

    def __init__(self, exog):
        self.exog = exog

    def fit(self, alpha=0):
        """estimates the regularized inverse covariance using nodewise
        regression

        Parameters
        ----------
        alpha : scalar
            Regularizing constant
        """
        n, p = self.exog.shape
        nodewise_row_l = []
        nodewise_weight_l = []
        for idx in range(p):
            nodewise_row = _calc_nodewise_row(self.exog, idx, alpha)
            nodewise_row_l.append(nodewise_row)
            nodewise_weight = _calc_nodewise_weight(self.exog, nodewise_row, idx, alpha)
            nodewise_weight_l.append(nodewise_weight)
        nodewise_row_l = np.array(nodewise_row_l)
        nodewise_weight_l = np.array(nodewise_weight_l)
        approx_inv_cov = _calc_approx_inv_cov(nodewise_row_l, nodewise_weight_l)
        self._approx_inv_cov = approx_inv_cov

    def approx_inv_cov(self):
        return self._approx_inv_cov