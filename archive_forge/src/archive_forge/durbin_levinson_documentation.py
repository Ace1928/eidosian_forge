from statsmodels.compat.pandas import deprecate_kwarg
import numpy as np
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.stattools import acovf

    Estimate AR parameters at multiple orders using Durbin-Levinson recursions.

    Parameters
    ----------
    endog : array_like or SARIMAXSpecification
        Input time series array, assumed to be stationary.
    ar_order : int, optional
        Autoregressive order. Default is 0.
    demean : bool, optional
        Whether to estimate and remove the mean from the process prior to
        fitting the autoregressive coefficients. Default is True.
    adjusted : bool, optional
        Whether to use the "adjusted" autocovariance estimator, which uses
        n - h degrees of freedom rather than n. This option can result in
        a non-positive definite autocovariance matrix. Default is False.

    Returns
    -------
    parameters : list of SARIMAXParams objects
        List elements correspond to estimates at different `ar_order`. For
        example, parameters[0] is an `SARIMAXParams` instance corresponding to
        `ar_order=0`.
    other_results : Bunch
        Includes one component, `spec`, containing the `SARIMAXSpecification`
        instance corresponding to the input arguments.

    Notes
    -----
    The primary reference is [1]_, section 2.5.1.

    This procedure assumes that the series is stationary.

    References
    ----------
    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.
       Introduction to Time Series and Forecasting. Springer.
    