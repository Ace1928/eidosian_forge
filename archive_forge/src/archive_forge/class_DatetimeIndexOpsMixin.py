from __future__ import annotations
from abc import (
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.arrays import (
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.tools.timedeltas import to_timedelta
class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex, ABC):
    """
    Common ops mixin to support a unified interface datetimelike Index.
    """
    _can_hold_strings = False
    _data: DatetimeArray | TimedeltaArray | PeriodArray

    @doc(DatetimeLikeArrayMixin.mean)
    def mean(self, *, skipna: bool=True, axis: int | None=0):
        return self._data.mean(skipna=skipna, axis=axis)

    @property
    def freq(self) -> BaseOffset | None:
        return self._data.freq

    @freq.setter
    def freq(self, value) -> None:
        self._data.freq = value

    @property
    def asi8(self) -> npt.NDArray[np.int64]:
        return self._data.asi8

    @property
    @doc(DatetimeLikeArrayMixin.freqstr)
    def freqstr(self) -> str:
        from pandas import PeriodIndex
        if self._data.freqstr is not None and isinstance(self._data, (PeriodArray, PeriodIndex)):
            freq = freq_to_period_freqstr(self._data.freq.n, self._data.freq.name)
            return freq
        else:
            return self._data.freqstr

    @cache_readonly
    @abstractmethod
    def _resolution_obj(self) -> Resolution:
        ...

    @cache_readonly
    @doc(DatetimeLikeArrayMixin.resolution)
    def resolution(self) -> str:
        return self._data.resolution

    @cache_readonly
    def hasnans(self) -> bool:
        return self._data._hasna

    def equals(self, other: Any) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
        if self.is_(other):
            return True
        if not isinstance(other, Index):
            return False
        elif other.dtype.kind in 'iufc':
            return False
        elif not isinstance(other, type(self)):
            should_try = False
            inferable = self._data._infer_matches
            if other.dtype == object:
                should_try = other.inferred_type in inferable
            elif isinstance(other.dtype, CategoricalDtype):
                other = cast('CategoricalIndex', other)
                should_try = other.categories.inferred_type in inferable
            if should_try:
                try:
                    other = type(self)(other)
                except (ValueError, TypeError, OverflowError):
                    return False
        if self.dtype != other.dtype:
            return False
        return np.array_equal(self.asi8, other.asi8)

    @Appender(Index.__contains__.__doc__)
    def __contains__(self, key: Any) -> bool:
        hash(key)
        try:
            self.get_loc(key)
        except (KeyError, TypeError, ValueError, InvalidIndexError):
            return False
        return True

    def _convert_tolerance(self, tolerance, target):
        tolerance = np.asarray(to_timedelta(tolerance).to_numpy())
        return super()._convert_tolerance(tolerance, target)
    _default_na_rep = 'NaT'

    def format(self, name: bool=False, formatter: Callable | None=None, na_rep: str='NaT', date_format: str | None=None) -> list[str]:
        """
        Render a string representation of the Index.
        """
        warnings.warn(f'{type(self).__name__}.format is deprecated and will be removed in a future version. Convert using index.astype(str) or index.map(formatter) instead.', FutureWarning, stacklevel=find_stack_level())
        header = []
        if name:
            header.append(ibase.pprint_thing(self.name, escape_chars=('\t', '\r', '\n')) if self.name is not None else '')
        if formatter is not None:
            return header + list(self.map(formatter))
        return self._format_with_header(header=header, na_rep=na_rep, date_format=date_format)

    def _format_with_header(self, *, header: list[str], na_rep: str, date_format: str | None=None) -> list[str]:
        return header + list(self._get_values_for_csv(na_rep=na_rep, date_format=date_format))

    @property
    def _formatter_func(self):
        return self._data._formatter()

    def _format_attrs(self):
        """
        Return a list of tuples of the (attr,formatted_value).
        """
        attrs = super()._format_attrs()
        for attrib in self._attributes:
            if attrib == 'freq':
                freq = self.freqstr
                if freq is not None:
                    freq = repr(freq)
                attrs.append(('freq', freq))
        return attrs

    @Appender(Index._summary.__doc__)
    def _summary(self, name=None) -> str:
        result = super()._summary(name=name)
        if self.freq:
            result += f'\nFreq: {self.freqstr}'
        return result

    @final
    def _can_partial_date_slice(self, reso: Resolution) -> bool:
        return reso > self._resolution_obj

    def _parsed_string_to_bounds(self, reso: Resolution, parsed):
        raise NotImplementedError

    def _parse_with_reso(self, label: str):
        try:
            if self.freq is None or hasattr(self.freq, 'rule_code'):
                freq = self.freq
        except NotImplementedError:
            freq = getattr(self, 'freqstr', getattr(self, 'inferred_freq', None))
        freqstr: str | None
        if freq is not None and (not isinstance(freq, str)):
            freqstr = freq.rule_code
        else:
            freqstr = freq
        if isinstance(label, np.str_):
            label = str(label)
        parsed, reso_str = parsing.parse_datetime_string_with_reso(label, freqstr)
        reso = Resolution.from_attrname(reso_str)
        return (parsed, reso)

    def _get_string_slice(self, key: str):
        parsed, reso = self._parse_with_reso(key)
        try:
            return self._partial_date_slice(reso, parsed)
        except KeyError as err:
            raise KeyError(key) from err

    @final
    def _partial_date_slice(self, reso: Resolution, parsed: datetime) -> slice | npt.NDArray[np.intp]:
        """
        Parameters
        ----------
        reso : Resolution
        parsed : datetime

        Returns
        -------
        slice or ndarray[intp]
        """
        if not self._can_partial_date_slice(reso):
            raise ValueError
        t1, t2 = self._parsed_string_to_bounds(reso, parsed)
        vals = self._data._ndarray
        unbox = self._data._unbox
        if self.is_monotonic_increasing:
            if len(self) and (t1 < self[0] and t2 < self[0] or (t1 > self[-1] and t2 > self[-1])):
                raise KeyError
            left = vals.searchsorted(unbox(t1), side='left')
            right = vals.searchsorted(unbox(t2), side='right')
            return slice(left, right)
        else:
            lhs_mask = vals >= unbox(t1)
            rhs_mask = vals <= unbox(t2)
            return (lhs_mask & rhs_mask).nonzero()[0]

    def _maybe_cast_slice_bound(self, label, side: str):
        """
        If label is a string, cast it to scalar type according to resolution.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}

        Returns
        -------
        label : object

        Notes
        -----
        Value of `side` parameter should be validated in caller.
        """
        if isinstance(label, str):
            try:
                parsed, reso = self._parse_with_reso(label)
            except ValueError as err:
                self._raise_invalid_indexer('slice', label, err)
            lower, upper = self._parsed_string_to_bounds(reso, parsed)
            return lower if side == 'left' else upper
        elif not isinstance(label, self._data._recognized_scalars):
            self._raise_invalid_indexer('slice', label)
        return label

    def shift(self, periods: int=1, freq=None) -> Self:
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or string, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.DatetimeIndex
            Shifted index.

        See Also
        --------
        Index.shift : Shift values of Index.
        PeriodIndex.shift : Shift values of PeriodIndex.
        """
        raise NotImplementedError

    @doc(Index._maybe_cast_listlike_indexer)
    def _maybe_cast_listlike_indexer(self, keyarr):
        try:
            res = self._data._validate_listlike(keyarr, allow_object=True)
        except (ValueError, TypeError):
            if not isinstance(keyarr, ExtensionArray):
                res = com.asarray_tuplesafe(keyarr)
            else:
                res = keyarr
        return Index(res, dtype=res.dtype)