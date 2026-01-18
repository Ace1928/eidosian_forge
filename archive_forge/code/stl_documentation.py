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
        