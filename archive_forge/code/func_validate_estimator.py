import numpy as np
import pandas as pd
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.arima.tools import standardize_lag_order, validate_basic
def validate_estimator(self, estimator):
    """
        Validate an SARIMA estimator.

        Parameters
        ----------
        estimator : str
            Name of the estimator to validate against the current state of
            the specification. Possible values are: 'yule_walker', 'burg',
            'innovations', 'hannan_rissanen', 'innovoations_mle', 'statespace'.

        Notes
        -----
        This method will raise a `ValueError` if an invalid method is passed,
        and otherwise will return None.

        This method does not consider the presense of `exog` in determining
        valid estimators. If there are exogenous variables, then feasible
        Generalized Least Squares should be used through the `gls` estimator,
        and a "valid" estimator is one that could be passed as the
        `arma_estimator` argument to `gls`.

        This method only uses the attributes `enforce_stationarity` and
        `concentrate_scale` to determine the validity of numerical maximum
        likelihood estimators. These only include 'innovations_mle' (which
        does not support `enforce_stationarity=False` or
        `concentrate_scale=True`) and 'statespace' (which supports all
        combinations of each).

        Examples
        --------
        >>> spec = SARIMAXSpecification(order=(1, 0, 2))

        >>> spec.validate_estimator('yule_walker')
        ValueError: Yule-Walker estimator does not support moving average
                    components.

        >>> spec.validate_estimator('burg')
        ValueError: Burg estimator does not support moving average components.

        >>> spec.validate_estimator('innovations')
        ValueError: Burg estimator does not support autoregressive components.

        >>> spec.validate_estimator('hannan_rissanen')  # returns None
        >>> spec.validate_estimator('innovations_mle')  # returns None
        >>> spec.validate_estimator('statespace')       # returns None

        >>> spec.validate_estimator('not_an_estimator')
        ValueError: "not_an_estimator" is not a valid estimator.
        """
    has_ar = self.max_ar_order != 0
    has_ma = self.max_ma_order != 0
    has_seasonal = self.seasonal_periods != 0
    has_missing = self._has_missing
    titles = {'yule_walker': 'Yule-Walker', 'burg': 'Burg', 'innovations': 'Innovations', 'hannan_rissanen': 'Hannan-Rissanen', 'innovations_mle': 'Innovations MLE', 'statespace': 'State space'}
    if estimator != 'statespace':
        if has_missing:
            raise ValueError('%s estimator does not support missing values in `endog`.' % titles[estimator])
    if estimator not in ['innovations_mle', 'statespace']:
        if self.max_ar_order > 0 and self.enforce_stationarity:
            raise ValueError('%s estimator cannot enforce a stationary autoregressive lag polynomial.' % titles[estimator])
        if self.max_ma_order > 0 and self.enforce_invertibility:
            raise ValueError('%s estimator cannot enforce an invertible moving average lag polynomial.' % titles[estimator])
    if estimator in ['yule_walker', 'burg']:
        if has_seasonal:
            raise ValueError('%s estimator does not support seasonal components.' % titles[estimator])
        if not self.is_ar_consecutive:
            raise ValueError('%s estimator does not support non-consecutive autoregressive lags.' % titles[estimator])
        if has_ma:
            raise ValueError('%s estimator does not support moving average components.' % titles[estimator])
    elif estimator == 'innovations':
        if has_seasonal:
            raise ValueError('Innovations estimator does not support seasonal components.')
        if not self.is_ma_consecutive:
            raise ValueError('Innovations estimator does not support non-consecutive moving average lags.')
        if has_ar:
            raise ValueError('Innovations estimator does not support autoregressive components.')
    elif estimator == 'hannan_rissanen':
        if has_seasonal:
            raise ValueError('Hannan-Rissanen estimator does not support seasonal components.')
    elif estimator == 'innovations_mle':
        if self.enforce_stationarity is False:
            raise ValueError('Innovations MLE estimator does not support non-stationary autoregressive components, but `enforce_stationarity` is set to False')
        if self.concentrate_scale:
            raise ValueError('Innovations MLE estimator does not support concentrating the scale out of the log-likelihood function')
    elif estimator == 'statespace':
        pass
    else:
        raise ValueError('"%s" is not a valid estimator.' % estimator)