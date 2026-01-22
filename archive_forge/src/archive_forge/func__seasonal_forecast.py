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