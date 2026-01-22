from __future__ import annotations
from statsmodels.compat.pandas import (
import numbers
import warnings
import numpy as np
from pandas import (
from pandas.tseries.frequencies import to_offset
from statsmodels.base.data import PandasData
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ValueWarning
def _init_dates(self, dates=None, freq=None):
    """
        Initialize dates

        Parameters
        ----------
        dates : array_like, optional
            An array like object containing dates.
        freq : str, tuple, datetime.timedelta, DateOffset or None, optional
            A frequency specification for either `dates` or the row labels from
            the endog / exog data.

        Notes
        -----
        Creates `self._index` and related attributes. `self._index` is always
        a Pandas index, and it is always NumericIndex, DatetimeIndex, or
        PeriodIndex.

        If Pandas objects, endog / exog may have any type of index. If it is
        an NumericIndex with values 0, 1, ..., nobs-1 or if it is (coerceable to)
        a DatetimeIndex or PeriodIndex *with an associated frequency*, then it
        is called a "supported" index. Otherwise it is called an "unsupported"
        index.

        Supported indexes are standardized (i.e. a list of date strings is
        converted to a DatetimeIndex) and the result is put in `self._index`.

        Unsupported indexes are ignored, and a supported NumericIndex is
        generated and put in `self._index`. Warnings are issued in this case
        to alert the user if the returned index from some operation (e.g.
        forecasting) is different from the original data's index. However,
        whenever possible (e.g. purely in-sample prediction), the original
        index is returned.

        The benefit of supported indexes is that they allow *forecasting*, i.e.
        it is possible to extend them in a reasonable way. Thus every model
        must have an underlying supported index, even if it is just a generated
        NumericIndex.
        """
    if dates is not None:
        index = dates
    else:
        index = self.data.row_labels
    if index is None and freq is not None:
        raise ValueError('Frequency provided without associated index.')
    inferred_freq = False
    if index is not None:
        if not isinstance(index, (DatetimeIndex, PeriodIndex)):
            try:
                _index = np.asarray(index)
                if is_numeric_dtype(_index) or is_float_index(index) or isinstance(_index[0], float):
                    raise ValueError('Numeric index given')
                if isinstance(index, Series):
                    index = index.values
                _index = to_datetime(index)
                if not isinstance(_index, Index):
                    raise ValueError('Could not coerce to date index')
                index = _index
            except:
                if dates is not None:
                    raise ValueError('Non-date index index provided to `dates` argument.')
        if isinstance(index, (DatetimeIndex, PeriodIndex)):
            if freq is None and index.freq is None:
                freq = index.inferred_freq
                if freq is not None:
                    inferred_freq = True
                    if freq is not None:
                        warnings.warn('No frequency information was provided, so inferred frequency %s will be used.' % freq, ValueWarning, stacklevel=2)
            if freq is not None:
                freq = to_offset(freq)
            if freq is None and index.freq is None:
                if dates is not None:
                    raise ValueError('No frequency information was provided with date index and no frequency could be inferred.')
            elif freq is not None and index.freq is None:
                resampled_index = date_range(start=index[0], end=index[-1], freq=freq)
                if not inferred_freq and (not resampled_index.equals(index)):
                    raise ValueError('The given frequency argument could not be matched to the given index.')
                index = resampled_index
            elif freq is not None and (not inferred_freq) and (not index.freq == freq):
                raise ValueError('The given frequency argument is incompatible with the given index.')
        elif freq is not None:
            raise ValueError('Given index could not be coerced to dates but `freq` argument was provided.')
    has_index = index is not None
    date_index = isinstance(index, (DatetimeIndex, PeriodIndex))
    period_index = isinstance(index, PeriodIndex)
    int_index = is_int_index(index)
    range_index = isinstance(index, RangeIndex)
    has_freq = index.freq is not None if date_index else None
    increment = Index(range(self.endog.shape[0]))
    is_increment = index.equals(increment) if int_index else None
    if date_index:
        try:
            is_monotonic = index.is_monotonic_increasing
        except AttributeError:
            is_monotonic = index.is_monotonic
    else:
        is_monotonic = None
    if has_index and (not (date_index or range_index or is_increment)):
        warnings.warn('An unsupported index was provided and will be ignored when e.g. forecasting.', ValueWarning, stacklevel=2)
    if date_index and (not has_freq):
        warnings.warn('A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.', ValueWarning, stacklevel=2)
    if date_index and (not is_monotonic):
        warnings.warn('A date index has been provided, but it is not monotonic and so will be ignored when e.g. forecasting.', ValueWarning, stacklevel=2)
    index_generated = False
    valid_index = date_index and has_freq and is_monotonic or (int_index and is_increment) or range_index
    if valid_index:
        _index = index
    else:
        _index = increment
        index_generated = True
    self._index = _index
    self._index_generated = index_generated
    self._index_none = index is None
    self._index_int64 = int_index and (not range_index) and (not date_index)
    self._index_dates = date_index and (not index_generated)
    self._index_freq = self._index.freq if self._index_dates else None
    self._index_inferred_freq = inferred_freq
    self.data.dates = self._index if self._index_dates else None
    self.data.freq = self._index.freqstr if self._index_dates else None