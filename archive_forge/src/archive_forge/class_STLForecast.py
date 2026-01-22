from statsmodels.compat.pandas import Substitution, is_int_index
import datetime as dt
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from statsmodels.base.data import PandasData
from statsmodels.iolib.summary import SimpleTable, Summary
from statsmodels.tools.docstring import Docstring, Parameter, indent
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.base.tsa_model import get_index_loc, get_prediction_index
from statsmodels.tsa.seasonal import STL, DecomposeResult
from statsmodels.tsa.statespace.kalman_filter import _check_dynamic
@Substitution(stl_forecast_params=indent(_stl_forecast_params, '    '))
class STLForecast:
    """
    Model-based forecasting using STL to remove seasonality

    Forecasts are produced by first subtracting the seasonality
    estimated using STL, then forecasting the deseasonalized
    data using a time-series model, for example, ARIMA.

    Parameters
    ----------
%(stl_forecast_params)s

    See Also
    --------
    statsmodels.tsa.arima.model.ARIMA
        ARIMA modeling.
    statsmodels.tsa.ar_model.AutoReg
        Autoregressive modeling supporting complex deterministics.
    statsmodels.tsa.exponential_smoothing.ets.ETSModel
        Additive and multiplicative exponential smoothing with trend.
    statsmodels.tsa.statespace.exponential_smoothing.ExponentialSmoothing
        Additive exponential smoothing with trend.

    Notes
    -----
    If :math:`\\hat{S}_t` is the seasonal component, then the deseasonalize
    series is constructed as

    .. math::

        Y_t - \\hat{S}_t

    The trend component is not removed, and so the time series model should
    be capable of adequately fitting and forecasting the trend if present. The
    out-of-sample forecasts of the seasonal component are produced as

    .. math::

        \\hat{S}_{T + h} = \\hat{S}_{T - k}

    where :math:`k = m - h + m \\lfloor (h-1)/m \\rfloor` tracks the period
    offset in the full cycle of 1, 2, ..., m where m is the period length.

    This class is mostly a convenience wrapper around ``STL`` and a
    user-specified model. The model is assumed to follow the standard
    statsmodels pattern:

    * ``fit`` is used to estimate parameters and returns a results instance,
      ``results``.
    * ``results`` must exposes a method ``forecast(steps, **kwargs)`` that
      produces out-of-sample forecasts.
    * ``results`` may also exposes a method ``get_prediction`` that produces
      both in- and out-of-sample predictions.

    See the notebook `Seasonal Decomposition
    <../examples/notebooks/generated/stl_decomposition.html>`__ for an
    overview.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from statsmodels.tsa.api import STLForecast
    >>> from statsmodels.tsa.arima.model import ARIMA
    >>> from statsmodels.datasets import macrodata
    >>> ds = macrodata.load_pandas()
    >>> data = np.log(ds.data.m1)
    >>> base_date = f"{int(ds.data.year[0])}-{3*int(ds.data.quarter[0])+1}-1"
    >>> data.index = pd.date_range(base_date, periods=data.shape[0], freq="QS")

    Generate forecasts from an ARIMA

    >>> stlf = STLForecast(data, ARIMA, model_kwargs={"order": (2, 1, 0)})
    >>> res = stlf.fit()
    >>> forecasts = res.forecast(12)

    Generate forecasts from an Exponential Smoothing model with trend

    >>> from statsmodels.tsa.statespace import exponential_smoothing
    >>> ES = exponential_smoothing.ExponentialSmoothing
    >>> config = {"trend": True}
    >>> stlf = STLForecast(data, ES, model_kwargs=config)
    >>> res = stlf.fit()
    >>> forecasts = res.forecast(12)
    """

    def __init__(self, endog, model, *, model_kwargs=None, period=None, seasonal=7, trend=None, low_pass=None, seasonal_deg=1, trend_deg=1, low_pass_deg=1, robust=False, seasonal_jump=1, trend_jump=1, low_pass_jump=1):
        self._endog = endog
        self._stl_kwargs = dict(period=period, seasonal=seasonal, trend=trend, low_pass=low_pass, seasonal_deg=seasonal_deg, trend_deg=trend_deg, low_pass_deg=low_pass_deg, robust=robust, seasonal_jump=seasonal_jump, trend_jump=trend_jump, low_pass_jump=low_pass_jump)
        self._model = model
        self._model_kwargs = {} if model_kwargs is None else model_kwargs
        if not hasattr(model, 'fit'):
            raise AttributeError('model must expose a ``fit``  method.')

    @Substitution(fit_params=indent(_fit_params, ' ' * 8))
    def fit(self, *, inner_iter=None, outer_iter=None, fit_kwargs=None):
        """
        Estimate STL and forecasting model parameters.

        Parameters
        ----------
%(fit_params)s
        fit_kwargs : dict[str, Any]
            Any additional keyword arguments to pass to ``model``'s ``fit``
            method when estimating the model on the decomposed residuals.

        Returns
        -------
        STLForecastResults
            Results with forecasting methods.
        """
        fit_kwargs = {} if fit_kwargs is None else fit_kwargs
        stl = STL(self._endog, **self._stl_kwargs)
        stl_fit: DecomposeResult = stl.fit(inner_iter=inner_iter, outer_iter=outer_iter)
        model_endog = stl_fit.trend + stl_fit.resid
        mod = self._model(model_endog, **self._model_kwargs)
        res = mod.fit(**fit_kwargs)
        if not hasattr(res, 'forecast'):
            raise AttributeError("The model's result must expose a ``forecast`` method.")
        return STLForecastResults(stl, stl_fit, mod, res, self._endog)