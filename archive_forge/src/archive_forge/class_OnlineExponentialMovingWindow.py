from __future__ import annotations
import datetime
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs.tslibs import Timedelta
import pandas._libs.window.aggregations as window_aggregations
from pandas.util._decorators import doc
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core import common
from pandas.core.arrays.datetimelike import dtype_to_unit
from pandas.core.indexers.objects import (
from pandas.core.util.numba_ import (
from pandas.core.window.common import zsqrt
from pandas.core.window.doc import (
from pandas.core.window.numba_ import (
from pandas.core.window.online import (
from pandas.core.window.rolling import (
class OnlineExponentialMovingWindow(ExponentialMovingWindow):

    def __init__(self, obj: NDFrame, com: float | None=None, span: float | None=None, halflife: float | TimedeltaConvertibleTypes | None=None, alpha: float | None=None, min_periods: int | None=0, adjust: bool=True, ignore_na: bool=False, axis: Axis=0, times: np.ndarray | NDFrame | None=None, engine: str='numba', engine_kwargs: dict[str, bool] | None=None, *, selection=None) -> None:
        if times is not None:
            raise NotImplementedError('times is not implemented with online operations.')
        super().__init__(obj=obj, com=com, span=span, halflife=halflife, alpha=alpha, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na, axis=axis, times=times, selection=selection)
        self._mean = EWMMeanState(self._com, self.adjust, self.ignore_na, self.axis, obj.shape)
        if maybe_use_numba(engine):
            self.engine = engine
            self.engine_kwargs = engine_kwargs
        else:
            raise ValueError("'numba' is the only supported engine")

    def reset(self) -> None:
        """
        Reset the state captured by `update` calls.
        """
        self._mean.reset()

    def aggregate(self, func, *args, **kwargs):
        raise NotImplementedError('aggregate is not implemented.')

    def std(self, bias: bool=False, *args, **kwargs):
        raise NotImplementedError('std is not implemented.')

    def corr(self, other: DataFrame | Series | None=None, pairwise: bool | None=None, numeric_only: bool=False):
        raise NotImplementedError('corr is not implemented.')

    def cov(self, other: DataFrame | Series | None=None, pairwise: bool | None=None, bias: bool=False, numeric_only: bool=False):
        raise NotImplementedError('cov is not implemented.')

    def var(self, bias: bool=False, numeric_only: bool=False):
        raise NotImplementedError('var is not implemented.')

    def mean(self, *args, update=None, update_times=None, **kwargs):
        """
        Calculate an online exponentially weighted mean.

        Parameters
        ----------
        update: DataFrame or Series, default None
            New values to continue calculating the
            exponentially weighted mean from the last values and weights.
            Values should be float64 dtype.

            ``update`` needs to be ``None`` the first time the
            exponentially weighted mean is calculated.

        update_times: Series or 1-D np.ndarray, default None
            New times to continue calculating the
            exponentially weighted mean from the last values and weights.
            If ``None``, values are assumed to be evenly spaced
            in time.
            This feature is currently unsupported.

        Returns
        -------
        DataFrame or Series

        Examples
        --------
        >>> df = pd.DataFrame({"a": range(5), "b": range(5, 10)})
        >>> online_ewm = df.head(2).ewm(0.5).online()
        >>> online_ewm.mean()
              a     b
        0  0.00  5.00
        1  0.75  5.75
        >>> online_ewm.mean(update=df.tail(3))
                  a         b
        2  1.615385  6.615385
        3  2.550000  7.550000
        4  3.520661  8.520661
        >>> online_ewm.reset()
        >>> online_ewm.mean()
              a     b
        0  0.00  5.00
        1  0.75  5.75
        """
        result_kwargs = {}
        is_frame = self._selected_obj.ndim == 2
        if update_times is not None:
            raise NotImplementedError('update_times is not implemented.')
        update_deltas = np.ones(max(self._selected_obj.shape[self.axis - 1] - 1, 0), dtype=np.float64)
        if update is not None:
            if self._mean.last_ewm is None:
                raise ValueError('Must call mean with update=None first before passing update')
            result_from = 1
            result_kwargs['index'] = update.index
            if is_frame:
                last_value = self._mean.last_ewm[np.newaxis, :]
                result_kwargs['columns'] = update.columns
            else:
                last_value = self._mean.last_ewm
                result_kwargs['name'] = update.name
            np_array = np.concatenate((last_value, update.to_numpy()))
        else:
            result_from = 0
            result_kwargs['index'] = self._selected_obj.index
            if is_frame:
                result_kwargs['columns'] = self._selected_obj.columns
            else:
                result_kwargs['name'] = self._selected_obj.name
            np_array = self._selected_obj.astype(np.float64, copy=False).to_numpy()
        ewma_func = generate_online_numba_ewma_func(**get_jit_arguments(self.engine_kwargs))
        result = self._mean.run_ewm(np_array if is_frame else np_array[:, np.newaxis], update_deltas, self.min_periods, ewma_func)
        if not is_frame:
            result = result.squeeze()
        result = result[result_from:]
        result = self._selected_obj._constructor(result, **result_kwargs)
        return result