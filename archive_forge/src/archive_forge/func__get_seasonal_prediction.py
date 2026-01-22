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