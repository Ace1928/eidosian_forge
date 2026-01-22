import numpy as np
from statsmodels.tools.validation import array_like
def aic_sigma(sigma2, nobs, df_modelwc, islog=False):
    """
    Akaike information criterion

    Parameters
    ----------
    sigma2 : float
        estimate of the residual variance or determinant of Sigma_hat in the
        multivariate case. If islog is true, then it is assumed that sigma
        is already log-ed, for example logdetSigma.
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    aic : float
        information criterion

    Notes
    -----
    A constant has been dropped in comparison to the loglikelihood base
    information criteria. The information criteria should be used to compare
    only comparable models.

    For example, AIC is defined in terms of the loglikelihood as

    :math:`-2 llf + 2 k`

    in terms of :math:`\\hat{\\sigma}^2`

    :math:`log(\\hat{\\sigma}^2) + 2 k / n`

    in terms of the determinant of :math:`\\hat{\\Sigma}`

    :math:`log(\\|\\hat{\\Sigma}\\|) + 2 k / n`

    Note: In our definition we do not divide by n in the log-likelihood
    version.

    TODO: Latex math

    reference for example lecture notes by Herman Bierens

    See Also
    --------

    References
    ----------
    https://en.wikipedia.org/wiki/Akaike_information_criterion
    """
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + aic(0, nobs, df_modelwc) / nobs