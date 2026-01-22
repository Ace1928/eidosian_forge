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
class STLForecastResults:
    """
    Results for forecasting using STL to remove seasonality

    Parameters
    ----------
    stl : STL
        The STL instance used to decompose the data.
    result : DecomposeResult
        The result of applying STL to the data.
    model : Model
        The time series model used to model the non-seasonal dynamics.
    model_result : Results
        Model results instance supporting, at a minimum, ``forecast``.
    """

    def __init__(self, stl: STL, result: DecomposeResult, model, model_result, endog) -> None:
        self._stl = stl
        self._result = result
        self._model = model
        self._model_result = model_result
        self._endog = np.asarray(endog)
        self._nobs = self._endog.shape[0]
        self._index = getattr(endog, 'index', pd.RangeIndex(self._nobs))
        if not (isinstance(self._index, (pd.DatetimeIndex, pd.PeriodIndex)) or is_int_index(self._index)):
            try:
                self._index = pd.to_datetime(self._index)
            except ValueError:
                self._index = pd.RangeIndex(self._nobs)

    @property
    def period(self) -> int:
        """The period of the seasonal component"""
        return self._stl.period

    @property
    def stl(self) -> STL:
        """The STL instance used to decompose the time series"""
        return self._stl

    @property
    def result(self) -> DecomposeResult:
        """The result of applying STL to the data"""
        return self._result

    @property
    def model(self) -> Any:
        """The model fit to the additively deseasonalized data"""
        return self._model

    @property
    def model_result(self) -> Any:
        """The result class from the estimated model"""
        return self._model_result

    def summary(self) -> Summary:
        """
        Summary of both the STL decomposition and the model fit.

        Returns
        -------
        Summary
            The summary of the model fit and the STL decomposition.

        Notes
        -----
        Requires that the model's result class supports ``summary`` and
        returns a ``Summary`` object.
        """
        if not hasattr(self._model_result, 'summary'):
            raise AttributeError('The model result does not have a summary attribute.')
        summary: Summary = self._model_result.summary()
        if not isinstance(summary, Summary):
            raise TypeError("The model result's summary is not a Summary object.")
        summary.tables[0].title = 'STL Decomposition and ' + summary.tables[0].title
        config = self._stl.config
        left_keys = ('period', 'seasonal', 'robust')
        left_data = []
        left_stubs = []
        right_data = []
        right_stubs = []
        for key in config:
            new = key.capitalize()
            new = new.replace('_', ' ')
            if new in ('Trend', 'Low Pass'):
                new += ' Length'
            is_left = any((key.startswith(val) for val in left_keys))
            new += ':'
            stub = f'{new:<23s}'
            val = f'{str(config[key]):>13s}'
            if is_left:
                left_stubs.append(stub)
                left_data.append([val])
            else:
                right_stubs.append(' ' * 6 + stub)
                right_data.append([val])
        tab = SimpleTable(left_data, stubs=tuple(left_stubs), title='STL Configuration')
        tab.extend_right(SimpleTable(right_data, stubs=right_stubs))
        summary.tables.append(tab)
        return summary

    def _get_seasonal_prediction(self, start: Optional[DateLike], end: Optional[DateLike], dynamic: Union[bool, DateLike]) -> np.ndarray:
        """
        Get STLs seasonal in- and out-of-sample predictions

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        dynamic : bool, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.

        Returns
        -------
        ndarray
            Array containing the seasibak predictions.
        """
        data = PandasData(pd.Series(self._endog), index=self._index)
        if start is None:
            start = 0
        start, end, out_of_sample, prediction_index = get_prediction_index(start, end, self._nobs, self._index, data=data)
        if isinstance(dynamic, (str, dt.datetime, pd.Timestamp)):
            dynamic, _, _ = get_index_loc(dynamic, self._index)
            dynamic = dynamic - start
        elif dynamic is True:
            dynamic = 0
        elif dynamic is False:
            dynamic = None
        nobs = self._nobs
        dynamic, _ = _check_dynamic(dynamic, start, end, nobs)
        in_sample_end = end + 1 if dynamic is None else dynamic
        seasonal = np.asarray(self._result.seasonal)
        predictions = seasonal[start:in_sample_end]
        oos = np.empty((0,))
        if dynamic is not None:
            num = out_of_sample + end + 1 - dynamic
            oos = self._seasonal_forecast(num, None, offset=dynamic)
        elif out_of_sample:
            oos = self._seasonal_forecast(out_of_sample, None)
            oos_start = max(start - nobs, 0)
            oos = oos[oos_start:]
        predictions = np.r_[predictions, oos]
        return predictions

    def _seasonal_forecast(self, steps: int, index: Optional[pd.Index], offset=None) -> Union[pd.Series, np.ndarray]:
        """
        Get the seasonal component of the forecast

        Parameters
        ----------
        steps : int
            The number of steps required.
        index : pd.Index
            A pandas index to use. If None, returns an ndarray.
        offset : int
            The index of the first out-of-sample observation. If None, uses
            nobs.

        Returns
        -------
        seasonal : {ndarray, Series}
            The seasonal component.
        """
        period = self.period
        seasonal = np.asarray(self._result.seasonal)
        offset = self._nobs if offset is None else offset
        seasonal = seasonal[offset - period:offset]
        seasonal = np.tile(seasonal, steps // period + (steps % period != 0))
        seasonal = seasonal[:steps]
        if index is not None:
            seasonal = pd.Series(seasonal, index=index)
        return seasonal

    def forecast(self, steps: int=1, **kwargs: dict[str, Any]) -> Union[np.ndarray, pd.Series]:
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int, str, or datetime, optional
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency, steps
            must be an integer. Default
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. These arguments are passed into the time series
            model results' ``forecast`` method.

        Returns
        -------
        forecast : {ndarray, Series}
            Out of sample forecasts
        """
        forecast = self._model_result.forecast(steps=steps, **kwargs)
        index = forecast.index if isinstance(forecast, pd.Series) else None
        return forecast + self._seasonal_forecast(steps, index)

    def get_prediction(self, start: Optional[DateLike]=None, end: Optional[DateLike]=None, dynamic: Union[bool, DateLike]=False, **kwargs: dict[str, Any]):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        dynamic : bool, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. These arguments are passed into the time series
            model results' ``get_prediction`` method.

        Returns
        -------
        PredictionResults
            PredictionResults instance containing in-sample predictions,
            out-of-sample forecasts, and prediction intervals.
        """
        pred = self._model_result.get_prediction(start=start, end=end, dynamic=dynamic, **kwargs)
        seasonal_prediction = self._get_seasonal_prediction(start, end, dynamic)
        mean = pred.predicted_mean + seasonal_prediction
        try:
            var_pred_mean = pred.var_pred_mean
        except (AttributeError, NotImplementedError):
            import warnings
            warnings.warn(f'The variance of the predicted mean is not available using the {self.model.__class__.__name__} model class.', UserWarning, stacklevel=2)
            var_pred_mean = np.nan + mean.copy()
        return PredictionResults(mean, var_pred_mean, dist='norm', row_labels=pred.row_labels)