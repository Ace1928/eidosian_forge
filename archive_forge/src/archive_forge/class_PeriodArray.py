from __future__ import annotations
from datetime import timedelta
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.fields import isleapyear_arr
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import datetimelike as dtl
import pandas.core.common as com
class PeriodArray(dtl.DatelikeOps, libperiod.PeriodMixin):
    """
    Pandas ExtensionArray for storing Period data.

    Users should use :func:`~pandas.array` to create new instances.

    Parameters
    ----------
    values : Union[PeriodArray, Series[period], ndarray[int], PeriodIndex]
        The data to store. These should be arrays that can be directly
        converted to ordinals without inference or copy (PeriodArray,
        ndarray[int64]), or a box around such an array (Series[period],
        PeriodIndex).
    dtype : PeriodDtype, optional
        A PeriodDtype instance from which to extract a `freq`. If both
        `freq` and `dtype` are specified, then the frequencies must match.
    freq : str or DateOffset
        The `freq` to use for the array. Mostly applicable when `values`
        is an ndarray of integers, when `freq` is required. When `values`
        is a PeriodArray (or box around), it's checked that ``values.freq``
        matches `freq`.
    copy : bool, default False
        Whether to copy the ordinals before storing.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    Period: Represents a period of time.
    PeriodIndex : Immutable Index for period data.
    period_range: Create a fixed-frequency PeriodArray.
    array: Construct a pandas array.

    Notes
    -----
    There are two components to a PeriodArray

    - ordinals : integer ndarray
    - freq : pd.tseries.offsets.Offset

    The values are physically stored as a 1-D ndarray of integers. These are
    called "ordinals" and represent some kind of offset from a base.

    The `freq` indicates the span covered by each element of the array.
    All elements in the PeriodArray have the same `freq`.

    Examples
    --------
    >>> pd.arrays.PeriodArray(pd.PeriodIndex(['2023-01-01',
    ...                                       '2023-01-02'], freq='D'))
    <PeriodArray>
    ['2023-01-01', '2023-01-02']
    Length: 2, dtype: period[D]
    """
    __array_priority__ = 1000
    _typ = 'periodarray'
    _internal_fill_value = np.int64(iNaT)
    _recognized_scalars = (Period,)
    _is_recognized_dtype = lambda x: isinstance(x, PeriodDtype)
    _infer_matches = ('period',)

    @property
    def _scalar_type(self) -> type[Period]:
        return Period
    _other_ops: list[str] = []
    _bool_ops: list[str] = ['is_leap_year']
    _object_ops: list[str] = ['start_time', 'end_time', 'freq']
    _field_ops: list[str] = ['year', 'month', 'day', 'hour', 'minute', 'second', 'weekofyear', 'weekday', 'week', 'dayofweek', 'day_of_week', 'dayofyear', 'day_of_year', 'quarter', 'qyear', 'days_in_month', 'daysinmonth']
    _datetimelike_ops: list[str] = _field_ops + _object_ops + _bool_ops
    _datetimelike_methods: list[str] = ['strftime', 'to_timestamp', 'asfreq']
    _dtype: PeriodDtype

    def __init__(self, values, dtype: Dtype | None=None, freq=None, copy: bool=False) -> None:
        if freq is not None:
            warnings.warn("The 'freq' keyword in the PeriodArray constructor is deprecated and will be removed in a future version. Pass 'dtype' instead", FutureWarning, stacklevel=find_stack_level())
            freq = validate_dtype_freq(dtype, freq)
            dtype = PeriodDtype(freq)
        if dtype is not None:
            dtype = pandas_dtype(dtype)
            if not isinstance(dtype, PeriodDtype):
                raise ValueError(f'Invalid dtype {dtype} for PeriodArray')
        if isinstance(values, ABCSeries):
            values = values._values
            if not isinstance(values, type(self)):
                raise TypeError('Incorrect dtype')
        elif isinstance(values, ABCPeriodIndex):
            values = values._values
        if isinstance(values, type(self)):
            if dtype is not None and dtype != values.dtype:
                raise raise_on_incompatible(values, dtype.freq)
            values, dtype = (values._ndarray, values.dtype)
        values = np.array(values, dtype='int64', copy=copy)
        if dtype is None:
            raise ValueError('dtype is not specified and cannot be inferred')
        dtype = cast(PeriodDtype, dtype)
        NDArrayBacked.__init__(self, values, dtype)

    @classmethod
    def _simple_new(cls, values: npt.NDArray[np.int64], dtype: PeriodDtype) -> Self:
        assertion_msg = 'Should be numpy array of type i8'
        assert isinstance(values, np.ndarray) and values.dtype == 'i8', assertion_msg
        return cls(values, dtype=dtype)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None=None, copy: bool=False) -> Self:
        if dtype is not None:
            dtype = pandas_dtype(dtype)
        if dtype and isinstance(dtype, PeriodDtype):
            freq = dtype.freq
        else:
            freq = None
        if isinstance(scalars, cls):
            validate_dtype_freq(scalars.dtype, freq)
            if copy:
                scalars = scalars.copy()
            return scalars
        periods = np.asarray(scalars, dtype=object)
        freq = freq or libperiod.extract_freq(periods)
        ordinals = libperiod.extract_ordinals(periods, freq)
        dtype = PeriodDtype(freq)
        return cls(ordinals, dtype=dtype)

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype: Dtype | None=None, copy: bool=False) -> Self:
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @classmethod
    def _from_datetime64(cls, data, freq, tz=None) -> Self:
        """
        Construct a PeriodArray from a datetime64 array

        Parameters
        ----------
        data : ndarray[datetime64[ns], datetime64[ns, tz]]
        freq : str or Tick
        tz : tzinfo, optional

        Returns
        -------
        PeriodArray[freq]
        """
        if isinstance(freq, BaseOffset):
            freq = freq_to_period_freqstr(freq.n, freq.name)
        data, freq = dt64arr_to_periodarr(data, freq, tz)
        dtype = PeriodDtype(freq)
        return cls(data, dtype=dtype)

    @classmethod
    def _generate_range(cls, start, end, periods, freq):
        periods = dtl.validate_periods(periods)
        if freq is not None:
            freq = Period._maybe_convert_freq(freq)
        if start is not None or end is not None:
            subarr, freq = _get_ordinal_range(start, end, periods, freq)
        else:
            raise ValueError('Not enough parameters to construct Period range')
        return (subarr, freq)

    @classmethod
    def _from_fields(cls, *, fields: dict, freq) -> Self:
        subarr, freq = _range_from_fields(freq=freq, **fields)
        dtype = PeriodDtype(freq)
        return cls._simple_new(subarr, dtype=dtype)

    def _unbox_scalar(self, value: Period | NaTType) -> np.int64:
        if value is NaT:
            return np.int64(value._value)
        elif isinstance(value, self._scalar_type):
            self._check_compatible_with(value)
            return np.int64(value.ordinal)
        else:
            raise ValueError(f"'value' should be a Period. Got '{value}' instead.")

    def _scalar_from_string(self, value: str) -> Period:
        return Period(value, freq=self.freq)

    def _check_compatible_with(self, other: Period | NaTType | PeriodArray) -> None:
        if other is NaT:
            return
        self._require_matching_freq(other.freq)

    @cache_readonly
    def dtype(self) -> PeriodDtype:
        return self._dtype

    @property
    def freq(self) -> BaseOffset:
        """
        Return the frequency object for this PeriodArray.
        """
        return self.dtype.freq

    @property
    def freqstr(self) -> str:
        return freq_to_period_freqstr(self.freq.n, self.freq.name)

    def __array__(self, dtype: NpDtype | None=None) -> np.ndarray:
        if dtype == 'i8':
            return self.asi8
        elif dtype == bool:
            return ~self._isnan
        return np.array(list(self), dtype=object)

    def __arrow_array__(self, type=None):
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow
        from pandas.core.arrays.arrow.extension_types import ArrowPeriodType
        if type is not None:
            if pyarrow.types.is_integer(type):
                return pyarrow.array(self._ndarray, mask=self.isna(), type=type)
            elif isinstance(type, ArrowPeriodType):
                if self.freqstr != type.freq:
                    raise TypeError(f"Not supported to convert PeriodArray to array with different 'freq' ({self.freqstr} vs {type.freq})")
            else:
                raise TypeError(f"Not supported to convert PeriodArray to '{type}' type")
        period_type = ArrowPeriodType(self.freqstr)
        storage_array = pyarrow.array(self._ndarray, mask=self.isna(), type='int64')
        return pyarrow.ExtensionArray.from_storage(period_type, storage_array)
    year = _field_accessor('year', '\n        The year of the period.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")\n        >>> idx.year\n        Index([2023, 2024, 2025], dtype=\'int64\')\n        ')
    month = _field_accessor('month', '\n        The month as January=1, December=12.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")\n        >>> idx.month\n        Index([1, 2, 3], dtype=\'int64\')\n        ')
    day = _field_accessor('day', "\n        The days of the period.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(['2020-01-31', '2020-02-28'], freq='D')\n        >>> idx.day\n        Index([31, 28], dtype='int64')\n        ")
    hour = _field_accessor('hour', '\n        The hour of the period.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023-01-01 10:00", "2023-01-01 11:00"], freq=\'h\')\n        >>> idx.hour\n        Index([10, 11], dtype=\'int64\')\n        ')
    minute = _field_accessor('minute', '\n        The minute of the period.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023-01-01 10:30:00",\n        ...                       "2023-01-01 11:50:00"], freq=\'min\')\n        >>> idx.minute\n        Index([30, 50], dtype=\'int64\')\n        ')
    second = _field_accessor('second', '\n        The second of the period.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023-01-01 10:00:30",\n        ...                       "2023-01-01 10:00:31"], freq=\'s\')\n        >>> idx.second\n        Index([30, 31], dtype=\'int64\')\n        ')
    weekofyear = _field_accessor('week', '\n        The week ordinal of the year.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")\n        >>> idx.week  # It can be written `weekofyear`\n        Index([5, 9, 13], dtype=\'int64\')\n        ')
    week = weekofyear
    day_of_week = _field_accessor('day_of_week', '\n        The day of the week with Monday=0, Sunday=6.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023-01-01", "2023-01-02", "2023-01-03"], freq="D")\n        >>> idx.weekday\n        Index([6, 0, 1], dtype=\'int64\')\n        ')
    dayofweek = day_of_week
    weekday = dayofweek
    dayofyear = day_of_year = _field_accessor('day_of_year', '\n        The ordinal day of the year.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023-01-10", "2023-02-01", "2023-03-01"], freq="D")\n        >>> idx.dayofyear\n        Index([10, 32, 60], dtype=\'int64\')\n\n        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")\n        >>> idx\n        PeriodIndex([\'2023\', \'2024\', \'2025\'], dtype=\'period[Y-DEC]\')\n        >>> idx.dayofyear\n        Index([365, 366, 365], dtype=\'int64\')\n        ')
    quarter = _field_accessor('quarter', '\n        The quarter of the date.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")\n        >>> idx.quarter\n        Index([1, 1, 1], dtype=\'int64\')\n        ')
    qyear = _field_accessor('qyear')
    days_in_month = _field_accessor('days_in_month', '\n        The number of days in the month.\n\n        Examples\n        --------\n        For Series:\n\n        >>> period = pd.period_range(\'2020-1-1 00:00\', \'2020-3-1 00:00\', freq=\'M\')\n        >>> s = pd.Series(period)\n        >>> s\n        0   2020-01\n        1   2020-02\n        2   2020-03\n        dtype: period[M]\n        >>> s.dt.days_in_month\n        0    31\n        1    29\n        2    31\n        dtype: int64\n\n        For PeriodIndex:\n\n        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")\n        >>> idx.days_in_month   # It can be also entered as `daysinmonth`\n        Index([31, 28, 31], dtype=\'int64\')\n        ')
    daysinmonth = days_in_month

    @property
    def is_leap_year(self) -> npt.NDArray[np.bool_]:
        """
        Logical indicating if the date belongs to a leap year.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")
        >>> idx.is_leap_year
        array([False,  True, False])
        """
        return isleapyear_arr(np.asarray(self.year))

    def to_timestamp(self, freq=None, how: str='start') -> DatetimeArray:
        """
        Cast to DatetimeArray/Index.

        Parameters
        ----------
        freq : str or DateOffset, optional
            Target frequency. The default is 'D' for week or longer,
            's' otherwise.
        how : {'s', 'e', 'start', 'end'}
            Whether to use the start or end of the time period being converted.

        Returns
        -------
        DatetimeArray/Index

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.to_timestamp()
        DatetimeIndex(['2023-01-01', '2023-02-01', '2023-03-01'],
        dtype='datetime64[ns]', freq='MS')
        """
        from pandas.core.arrays import DatetimeArray
        how = libperiod.validate_end_alias(how)
        end = how == 'E'
        if end:
            if freq == 'B' or self.freq == 'B':
                adjust = Timedelta(1, 'D') - Timedelta(1, 'ns')
                return self.to_timestamp(how='start') + adjust
            else:
                adjust = Timedelta(1, 'ns')
                return (self + self.freq).to_timestamp(how='start') - adjust
        if freq is None:
            freq_code = self._dtype._get_to_timestamp_base()
            dtype = PeriodDtypeBase(freq_code, 1)
            freq = dtype._freqstr
            base = freq_code
        else:
            freq = Period._maybe_convert_freq(freq)
            base = freq._period_dtype_code
        new_parr = self.asfreq(freq, how=how)
        new_data = libperiod.periodarr_to_dt64arr(new_parr.asi8, base)
        dta = DatetimeArray._from_sequence(new_data)
        if self.freq.name == 'B':
            diffs = libalgos.unique_deltas(self.asi8)
            if len(diffs) == 1:
                diff = diffs[0]
                if diff == self.dtype._n:
                    dta._freq = self.freq
                elif diff == 1:
                    dta._freq = self.freq.base
            return dta
        else:
            return dta._with_freq('infer')

    def _box_func(self, x) -> Period | NaTType:
        return Period._from_ordinal(ordinal=x, freq=self.freq)

    @doc(**_shared_doc_kwargs, other='PeriodIndex', other_name='PeriodIndex')
    def asfreq(self, freq=None, how: str='E') -> Self:
        """
        Convert the {klass} to the specified frequency `freq`.

        Equivalent to applying :meth:`pandas.Period.asfreq` with the given arguments
        to each :class:`~pandas.Period` in this {klass}.

        Parameters
        ----------
        freq : str
            A frequency.
        how : str {{'E', 'S'}}, default 'E'
            Whether the elements should be aligned to the end
            or start within pa period.

            * 'E', 'END', or 'FINISH' for end,
            * 'S', 'START', or 'BEGIN' for start.

            January 31st ('END') vs. January 1st ('START') for example.

        Returns
        -------
        {klass}
            The transformed {klass} with the new frequency.

        See Also
        --------
        {other}.asfreq: Convert each Period in a {other_name} to the given frequency.
        Period.asfreq : Convert a :class:`~pandas.Period` object to the given frequency.

        Examples
        --------
        >>> pidx = pd.period_range('2010-01-01', '2015-01-01', freq='Y')
        >>> pidx
        PeriodIndex(['2010', '2011', '2012', '2013', '2014', '2015'],
        dtype='period[Y-DEC]')

        >>> pidx.asfreq('M')
        PeriodIndex(['2010-12', '2011-12', '2012-12', '2013-12', '2014-12',
        '2015-12'], dtype='period[M]')

        >>> pidx.asfreq('M', how='S')
        PeriodIndex(['2010-01', '2011-01', '2012-01', '2013-01', '2014-01',
        '2015-01'], dtype='period[M]')
        """
        how = libperiod.validate_end_alias(how)
        if isinstance(freq, BaseOffset) and hasattr(freq, '_period_dtype_code'):
            freq = PeriodDtype(freq)._freqstr
        freq = Period._maybe_convert_freq(freq)
        base1 = self._dtype._dtype_code
        base2 = freq._period_dtype_code
        asi8 = self.asi8
        end = how == 'E'
        if end:
            ordinal = asi8 + self.dtype._n - 1
        else:
            ordinal = asi8
        new_data = period_asfreq_arr(ordinal, base1, base2, end)
        if self._hasna:
            new_data[self._isnan] = iNaT
        dtype = PeriodDtype(freq)
        return type(self)(new_data, dtype=dtype)

    def _formatter(self, boxed: bool=False):
        if boxed:
            return str
        return "'{}'".format

    def _format_native_types(self, *, na_rep: str | float='NaT', date_format=None, **kwargs) -> npt.NDArray[np.object_]:
        """
        actually format my specific types
        """
        return libperiod.period_array_strftime(self.asi8, self.dtype._dtype_code, na_rep, date_format)

    def astype(self, dtype, copy: bool=True):
        dtype = pandas_dtype(dtype)
        if dtype == self._dtype:
            if not copy:
                return self
            else:
                return self.copy()
        if isinstance(dtype, PeriodDtype):
            return self.asfreq(dtype.freq)
        if lib.is_np_dtype(dtype, 'M') or isinstance(dtype, DatetimeTZDtype):
            tz = getattr(dtype, 'tz', None)
            unit = dtl.dtype_to_unit(dtype)
            return self.to_timestamp().tz_localize(tz).as_unit(unit)
        return super().astype(dtype, copy=copy)

    def searchsorted(self, value: NumpyValueArrayLike | ExtensionArray, side: Literal['left', 'right']='left', sorter: NumpySorter | None=None) -> npt.NDArray[np.intp] | np.intp:
        npvalue = self._validate_setitem_value(value).view('M8[ns]')
        m8arr = self._ndarray.view('M8[ns]')
        return m8arr.searchsorted(npvalue, side=side, sorter=sorter)

    def _pad_or_backfill(self, *, method: FillnaOptions, limit: int | None=None, limit_area: Literal['inside', 'outside'] | None=None, copy: bool=True) -> Self:
        dta = self.view('M8[ns]')
        result = dta._pad_or_backfill(method=method, limit=limit, limit_area=limit_area, copy=copy)
        if copy:
            return cast('Self', result.view(self.dtype))
        else:
            return self

    def fillna(self, value=None, method=None, limit: int | None=None, copy: bool=True) -> Self:
        if method is not None:
            dta = self.view('M8[ns]')
            result = dta.fillna(value=value, method=method, limit=limit, copy=copy)
            return result.view(self.dtype)
        return super().fillna(value=value, method=method, limit=limit, copy=copy)

    def _addsub_int_array_or_scalar(self, other: np.ndarray | int, op: Callable[[Any, Any], Any]) -> Self:
        """
        Add or subtract array of integers.

        Parameters
        ----------
        other : np.ndarray[int64] or int
        op : {operator.add, operator.sub}

        Returns
        -------
        result : PeriodArray
        """
        assert op in [operator.add, operator.sub]
        if op is operator.sub:
            other = -other
        res_values = add_overflowsafe(self.asi8, np.asarray(other, dtype='i8'))
        return type(self)(res_values, dtype=self.dtype)

    def _add_offset(self, other: BaseOffset):
        assert not isinstance(other, Tick)
        self._require_matching_freq(other, base=True)
        return self._addsub_int_array_or_scalar(other.n, operator.add)

    def _add_timedeltalike_scalar(self, other):
        """
        Parameters
        ----------
        other : timedelta, Tick, np.timedelta64

        Returns
        -------
        PeriodArray
        """
        if not isinstance(self.freq, Tick):
            raise raise_on_incompatible(self, other)
        if isna(other):
            return super()._add_timedeltalike_scalar(other)
        td = np.asarray(Timedelta(other).asm8)
        return self._add_timedelta_arraylike(td)

    def _add_timedelta_arraylike(self, other: TimedeltaArray | npt.NDArray[np.timedelta64]) -> Self:
        """
        Parameters
        ----------
        other : TimedeltaArray or ndarray[timedelta64]

        Returns
        -------
        PeriodArray
        """
        if not self.dtype._is_tick_like():
            raise TypeError(f'Cannot add or subtract timedelta64[ns] dtype from {self.dtype}')
        dtype = np.dtype(f'm8[{self.dtype._td64_unit}]')
        try:
            delta = astype_overflowsafe(np.asarray(other), dtype=dtype, copy=False, round_ok=False)
        except ValueError as err:
            raise IncompatibleFrequency("Cannot add/subtract timedelta-like from PeriodArray that is not an integer multiple of the PeriodArray's freq.") from err
        res_values = add_overflowsafe(self.asi8, np.asarray(delta.view('i8')))
        return type(self)(res_values, dtype=self.dtype)

    def _check_timedeltalike_freq_compat(self, other):
        """
        Arithmetic operations with timedelta-like scalars or array `other`
        are only valid if `other` is an integer multiple of `self.freq`.
        If the operation is valid, find that integer multiple.  Otherwise,
        raise because the operation is invalid.

        Parameters
        ----------
        other : timedelta, np.timedelta64, Tick,
                ndarray[timedelta64], TimedeltaArray, TimedeltaIndex

        Returns
        -------
        multiple : int or ndarray[int64]

        Raises
        ------
        IncompatibleFrequency
        """
        assert self.dtype._is_tick_like()
        dtype = np.dtype(f'm8[{self.dtype._td64_unit}]')
        if isinstance(other, (timedelta, np.timedelta64, Tick)):
            td = np.asarray(Timedelta(other).asm8)
        else:
            td = np.asarray(other)
        try:
            delta = astype_overflowsafe(td, dtype=dtype, copy=False, round_ok=False)
        except ValueError as err:
            raise raise_on_incompatible(self, other) from err
        delta = delta.view('i8')
        return lib.item_from_zerodim(delta)