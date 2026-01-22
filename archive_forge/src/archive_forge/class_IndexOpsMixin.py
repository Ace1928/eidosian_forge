from __future__ import annotations
import textwrap
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas._libs import lib
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import can_hold_element
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import DirNamesMixin
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.construction import (
class IndexOpsMixin(OpsMixin):
    """
    Common ops mixin to support a unified interface / docs for Series / Index
    """
    __array_priority__ = 1000
    _hidden_attrs: frozenset[str] = frozenset(['tolist'])

    @property
    def dtype(self) -> DtypeObj:
        raise AbstractMethodError(self)

    @property
    def _values(self) -> ExtensionArray | np.ndarray:
        raise AbstractMethodError(self)

    @final
    def transpose(self, *args, **kwargs) -> Self:
        """
        Return the transpose, which is by definition self.

        Returns
        -------
        %(klass)s
        """
        nv.validate_transpose(args, kwargs)
        return self
    T = property(transpose, doc="\n        Return the transpose, which is by definition self.\n\n        Examples\n        --------\n        For Series:\n\n        >>> s = pd.Series(['Ant', 'Bear', 'Cow'])\n        >>> s\n        0     Ant\n        1    Bear\n        2     Cow\n        dtype: object\n        >>> s.T\n        0     Ant\n        1    Bear\n        2     Cow\n        dtype: object\n\n        For Index:\n\n        >>> idx = pd.Index([1, 2, 3])\n        >>> idx.T\n        Index([1, 2, 3], dtype='int64')\n        ")

    @property
    def shape(self) -> Shape:
        """
        Return a tuple of the shape of the underlying data.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.shape
        (3,)
        """
        return self._values.shape

    def __len__(self) -> int:
        raise AbstractMethodError(self)

    @property
    def ndim(self) -> Literal[1]:
        """
        Number of dimensions of the underlying data, by definition 1.

        Examples
        --------
        >>> s = pd.Series(['Ant', 'Bear', 'Cow'])
        >>> s
        0     Ant
        1    Bear
        2     Cow
        dtype: object
        >>> s.ndim
        1

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.ndim
        1
        """
        return 1

    @final
    def item(self):
        """
        Return the first element of the underlying data as a Python scalar.

        Returns
        -------
        scalar
            The first element of Series or Index.

        Raises
        ------
        ValueError
            If the data is not length = 1.

        Examples
        --------
        >>> s = pd.Series([1])
        >>> s.item()
        1

        For an index:

        >>> s = pd.Series([1], index=['a'])
        >>> s.index.item()
        'a'
        """
        if len(self) == 1:
            return next(iter(self))
        raise ValueError('can only convert an array of size 1 to a Python scalar')

    @property
    def nbytes(self) -> int:
        """
        Return the number of bytes in the underlying data.

        Examples
        --------
        For Series:

        >>> s = pd.Series(['Ant', 'Bear', 'Cow'])
        >>> s
        0     Ant
        1    Bear
        2     Cow
        dtype: object
        >>> s.nbytes
        24

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.nbytes
        24
        """
        return self._values.nbytes

    @property
    def size(self) -> int:
        """
        Return the number of elements in the underlying data.

        Examples
        --------
        For Series:

        >>> s = pd.Series(['Ant', 'Bear', 'Cow'])
        >>> s
        0     Ant
        1    Bear
        2     Cow
        dtype: object
        >>> s.size
        3

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.size
        3
        """
        return len(self._values)

    @property
    def array(self) -> ExtensionArray:
        """
        The ExtensionArray of the data backing this Series or Index.

        Returns
        -------
        ExtensionArray
            An ExtensionArray of the values stored within. For extension
            types, this is the actual array. For NumPy native types, this
            is a thin (no copy) wrapper around :class:`numpy.ndarray`.

            ``.array`` differs from ``.values``, which may require converting
            the data to a different form.

        See Also
        --------
        Index.to_numpy : Similar method that always returns a NumPy array.
        Series.to_numpy : Similar method that always returns a NumPy array.

        Notes
        -----
        This table lays out the different array types for each extension
        dtype within pandas.

        ================== =============================
        dtype              array type
        ================== =============================
        category           Categorical
        period             PeriodArray
        interval           IntervalArray
        IntegerNA          IntegerArray
        string             StringArray
        boolean            BooleanArray
        datetime64[ns, tz] DatetimeArray
        ================== =============================

        For any 3rd-party extension types, the array type will be an
        ExtensionArray.

        For all remaining dtypes ``.array`` will be a
        :class:`arrays.NumpyExtensionArray` wrapping the actual ndarray
        stored within. If you absolutely need a NumPy array (possibly with
        copying / coercing data), then use :meth:`Series.to_numpy` instead.

        Examples
        --------
        For regular NumPy types like int, and float, a NumpyExtensionArray
        is returned.

        >>> pd.Series([1, 2, 3]).array
        <NumpyExtensionArray>
        [1, 2, 3]
        Length: 3, dtype: int64

        For extension types, like Categorical, the actual ExtensionArray
        is returned

        >>> ser = pd.Series(pd.Categorical(['a', 'b', 'a']))
        >>> ser.array
        ['a', 'b', 'a']
        Categories (2, object): ['a', 'b']
        """
        raise AbstractMethodError(self)

    @final
    def to_numpy(self, dtype: npt.DTypeLike | None=None, copy: bool=False, na_value: object=lib.no_default, **kwargs) -> np.ndarray:
        """
        A NumPy ndarray representing the values in this Series or Index.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.
        na_value : Any, optional
            The value to use for missing values. The default value depends
            on `dtype` and the type of the array.
        **kwargs
            Additional keywords passed through to the ``to_numpy`` method
            of the underlying array (for extension arrays).

        Returns
        -------
        numpy.ndarray

        See Also
        --------
        Series.array : Get the actual data stored within.
        Index.array : Get the actual data stored within.
        DataFrame.to_numpy : Similar method for DataFrame.

        Notes
        -----
        The returned array will be the same up to equality (values equal
        in `self` will be equal in the returned array; likewise for values
        that are not equal). When `self` contains an ExtensionArray, the
        dtype may be different. For example, for a category-dtype Series,
        ``to_numpy()`` will return a NumPy array and the categorical dtype
        will be lost.

        For NumPy dtypes, this will be a reference to the actual data stored
        in this Series or Index (assuming ``copy=False``). Modifying the result
        in place will modify the data stored in the Series or Index (not that
        we recommend doing that).

        For extension types, ``to_numpy()`` *may* require copying data and
        coercing the result to a NumPy type (possibly object), which may be
        expensive. When you need a no-copy reference to the underlying data,
        :attr:`Series.array` should be used instead.

        This table lays out the different dtypes and default return types of
        ``to_numpy()`` for various dtypes within pandas.

        ================== ================================
        dtype              array type
        ================== ================================
        category[T]        ndarray[T] (same dtype as input)
        period             ndarray[object] (Periods)
        interval           ndarray[object] (Intervals)
        IntegerNA          ndarray[object]
        datetime64[ns]     datetime64[ns]
        datetime64[ns, tz] ndarray[object] (Timestamps)
        ================== ================================

        Examples
        --------
        >>> ser = pd.Series(pd.Categorical(['a', 'b', 'a']))
        >>> ser.to_numpy()
        array(['a', 'b', 'a'], dtype=object)

        Specify the `dtype` to control how datetime-aware data is represented.
        Use ``dtype=object`` to return an ndarray of pandas :class:`Timestamp`
        objects, each with the correct ``tz``.

        >>> ser = pd.Series(pd.date_range('2000', periods=2, tz="CET"))
        >>> ser.to_numpy(dtype=object)
        array([Timestamp('2000-01-01 00:00:00+0100', tz='CET'),
               Timestamp('2000-01-02 00:00:00+0100', tz='CET')],
              dtype=object)

        Or ``dtype='datetime64[ns]'`` to return an ndarray of native
        datetime64 values. The values are converted to UTC and the timezone
        info is dropped.

        >>> ser.to_numpy(dtype="datetime64[ns]")
        ... # doctest: +ELLIPSIS
        array(['1999-12-31T23:00:00.000000000', '2000-01-01T23:00:00...'],
              dtype='datetime64[ns]')
        """
        if isinstance(self.dtype, ExtensionDtype):
            return self.array.to_numpy(dtype, copy=copy, na_value=na_value, **kwargs)
        elif kwargs:
            bad_keys = next(iter(kwargs.keys()))
            raise TypeError(f"to_numpy() got an unexpected keyword argument '{bad_keys}'")
        fillna = na_value is not lib.no_default and (not (na_value is np.nan and np.issubdtype(self.dtype, np.floating)))
        values = self._values
        if fillna:
            if not can_hold_element(values, na_value):
                values = np.asarray(values, dtype=dtype)
            else:
                values = values.copy()
            values[np.asanyarray(isna(self))] = na_value
        result = np.asarray(values, dtype=dtype)
        if copy and (not fillna) or (not copy and using_copy_on_write()):
            if np.shares_memory(self._values[:2], result[:2]):
                if using_copy_on_write() and (not copy):
                    result = result.view()
                    result.flags.writeable = False
                else:
                    result = result.copy()
        return result

    @final
    @property
    def empty(self) -> bool:
        return not self.size

    @doc(op='max', oppose='min', value='largest')
    def argmax(self, axis: AxisInt | None=None, skipna: bool=True, *args, **kwargs) -> int:
        """
        Return int position of the {value} value in the Series.

        If the {op}imum is achieved in multiple locations,
        the first row position is returned.

        Parameters
        ----------
        axis : {{None}}
            Unused. Parameter needed for compatibility with DataFrame.
        skipna : bool, default True
            Exclude NA/null values when showing the result.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        int
            Row position of the {op}imum value.

        See Also
        --------
        Series.arg{op} : Return position of the {op}imum value.
        Series.arg{oppose} : Return position of the {oppose}imum value.
        numpy.ndarray.arg{op} : Equivalent method for numpy arrays.
        Series.idxmax : Return index label of the maximum values.
        Series.idxmin : Return index label of the minimum values.

        Examples
        --------
        Consider dataset containing cereal calories

        >>> s = pd.Series({{'Corn Flakes': 100.0, 'Almond Delight': 110.0,
        ...                'Cinnamon Toast Crunch': 120.0, 'Cocoa Puff': 110.0}})
        >>> s
        Corn Flakes              100.0
        Almond Delight           110.0
        Cinnamon Toast Crunch    120.0
        Cocoa Puff               110.0
        dtype: float64

        >>> s.argmax()
        2
        >>> s.argmin()
        0

        The maximum cereal calories is the third element and
        the minimum cereal calories is the first element,
        since series is zero-indexed.
        """
        delegate = self._values
        nv.validate_minmax_axis(axis)
        skipna = nv.validate_argmax_with_skipna(skipna, args, kwargs)
        if isinstance(delegate, ExtensionArray):
            if not skipna and delegate.isna().any():
                warnings.warn(f'The behavior of {type(self).__name__}.argmax/argmin with skipna=False and NAs, or with all-NAs is deprecated. In a future version this will raise ValueError.', FutureWarning, stacklevel=find_stack_level())
                return -1
            else:
                return delegate.argmax()
        else:
            result = nanops.nanargmax(delegate, skipna=skipna)
            if result == -1:
                warnings.warn(f'The behavior of {type(self).__name__}.argmax/argmin with skipna=False and NAs, or with all-NAs is deprecated. In a future version this will raise ValueError.', FutureWarning, stacklevel=find_stack_level())
            return result

    @doc(argmax, op='min', oppose='max', value='smallest')
    def argmin(self, axis: AxisInt | None=None, skipna: bool=True, *args, **kwargs) -> int:
        delegate = self._values
        nv.validate_minmax_axis(axis)
        skipna = nv.validate_argmin_with_skipna(skipna, args, kwargs)
        if isinstance(delegate, ExtensionArray):
            if not skipna and delegate.isna().any():
                warnings.warn(f'The behavior of {type(self).__name__}.argmax/argmin with skipna=False and NAs, or with all-NAs is deprecated. In a future version this will raise ValueError.', FutureWarning, stacklevel=find_stack_level())
                return -1
            else:
                return delegate.argmin()
        else:
            result = nanops.nanargmin(delegate, skipna=skipna)
            if result == -1:
                warnings.warn(f'The behavior of {type(self).__name__}.argmax/argmin with skipna=False and NAs, or with all-NAs is deprecated. In a future version this will raise ValueError.', FutureWarning, stacklevel=find_stack_level())
            return result

    def tolist(self):
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        list

        See Also
        --------
        numpy.ndarray.tolist : Return the array as an a.ndim-levels deep
            nested list of Python scalars.

        Examples
        --------
        For Series

        >>> s = pd.Series([1, 2, 3])
        >>> s.to_list()
        [1, 2, 3]

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')

        >>> idx.to_list()
        [1, 2, 3]
        """
        return self._values.tolist()
    to_list = tolist

    def __iter__(self) -> Iterator:
        """
        Return an iterator of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        iterator

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> for x in s:
        ...     print(x)
        1
        2
        3
        """
        if not isinstance(self._values, np.ndarray):
            return iter(self._values)
        else:
            return map(self._values.item, range(self._values.size))

    @cache_readonly
    def hasnans(self) -> bool:
        """
        Return True if there are any NaNs.

        Enables various performance speedups.

        Returns
        -------
        bool

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, None])
        >>> s
        0    1.0
        1    2.0
        2    3.0
        3    NaN
        dtype: float64
        >>> s.hasnans
        True
        """
        return bool(isna(self).any())

    @final
    def _map_values(self, mapper, na_action=None, convert: bool=True):
        """
        An internal function that maps values using the input
        correspondence (which can be a dict, Series, or function).

        Parameters
        ----------
        mapper : function, dict, or Series
            The input correspondence object
        na_action : {None, 'ignore'}
            If 'ignore', propagate NA values, without passing them to the
            mapping function
        convert : bool, default True
            Try to find better dtype for elementwise function results. If
            False, leave as dtype=object. Note that the dtype is always
            preserved for some extension array dtypes, such as Categorical.

        Returns
        -------
        Union[Index, MultiIndex], inferred
            The output of the mapping function applied to the index.
            If the function returns a tuple with more than one element
            a MultiIndex will be returned.
        """
        arr = self._values
        if isinstance(arr, ExtensionArray):
            return arr.map(mapper, na_action=na_action)
        return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)

    @final
    def value_counts(self, normalize: bool=False, sort: bool=True, ascending: bool=False, bins=None, dropna: bool=True) -> Series:
        """
        Return a Series containing counts of unique values.

        The resulting object will be in descending order so that the
        first element is the most frequently-occurring element.
        Excludes NA values by default.

        Parameters
        ----------
        normalize : bool, default False
            If True then the object returned will contain the relative
            frequencies of the unique values.
        sort : bool, default True
            Sort by frequencies when True. Preserve the order of the data when False.
        ascending : bool, default False
            Sort in ascending order.
        bins : int, optional
            Rather than count values, group them into half-open bins,
            a convenience for ``pd.cut``, only works with numeric data.
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        Series

        See Also
        --------
        Series.count: Number of non-NA elements in a Series.
        DataFrame.count: Number of non-NA elements in a DataFrame.
        DataFrame.value_counts: Equivalent method on DataFrames.

        Examples
        --------
        >>> index = pd.Index([3, 1, 2, 3, 4, np.nan])
        >>> index.value_counts()
        3.0    2
        1.0    1
        2.0    1
        4.0    1
        Name: count, dtype: int64

        With `normalize` set to `True`, returns the relative frequency by
        dividing all values by the sum of values.

        >>> s = pd.Series([3, 1, 2, 3, 4, np.nan])
        >>> s.value_counts(normalize=True)
        3.0    0.4
        1.0    0.2
        2.0    0.2
        4.0    0.2
        Name: proportion, dtype: float64

        **bins**

        Bins can be useful for going from a continuous variable to a
        categorical variable; instead of counting unique
        apparitions of values, divide the index in the specified
        number of half-open bins.

        >>> s.value_counts(bins=3)
        (0.996, 2.0]    2
        (2.0, 3.0]      2
        (3.0, 4.0]      1
        Name: count, dtype: int64

        **dropna**

        With `dropna` set to `False` we can also see NaN index values.

        >>> s.value_counts(dropna=False)
        3.0    2
        1.0    1
        2.0    1
        4.0    1
        NaN    1
        Name: count, dtype: int64
        """
        return algorithms.value_counts_internal(self, sort=sort, ascending=ascending, normalize=normalize, bins=bins, dropna=dropna)

    def unique(self):
        values = self._values
        if not isinstance(values, np.ndarray):
            result = values.unique()
        else:
            result = algorithms.unique1d(values)
        return result

    @final
    def nunique(self, dropna: bool=True) -> int:
        """
        Return number of unique elements in the object.

        Excludes NA values by default.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the count.

        Returns
        -------
        int

        See Also
        --------
        DataFrame.nunique: Method nunique for DataFrame.
        Series.count: Count non-NA/null observations in the Series.

        Examples
        --------
        >>> s = pd.Series([1, 3, 5, 7, 7])
        >>> s
        0    1
        1    3
        2    5
        3    7
        4    7
        dtype: int64

        >>> s.nunique()
        4
        """
        uniqs = self.unique()
        if dropna:
            uniqs = remove_na_arraylike(uniqs)
        return len(uniqs)

    @property
    def is_unique(self) -> bool:
        """
        Return boolean if values in the object are unique.

        Returns
        -------
        bool

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.is_unique
        True

        >>> s = pd.Series([1, 2, 3, 1])
        >>> s.is_unique
        False
        """
        return self.nunique(dropna=False) == len(self)

    @property
    def is_monotonic_increasing(self) -> bool:
        """
        Return boolean if values in the object are monotonically increasing.

        Returns
        -------
        bool

        Examples
        --------
        >>> s = pd.Series([1, 2, 2])
        >>> s.is_monotonic_increasing
        True

        >>> s = pd.Series([3, 2, 1])
        >>> s.is_monotonic_increasing
        False
        """
        from pandas import Index
        return Index(self).is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self) -> bool:
        """
        Return boolean if values in the object are monotonically decreasing.

        Returns
        -------
        bool

        Examples
        --------
        >>> s = pd.Series([3, 2, 2, 1])
        >>> s.is_monotonic_decreasing
        True

        >>> s = pd.Series([1, 2, 3])
        >>> s.is_monotonic_decreasing
        False
        """
        from pandas import Index
        return Index(self).is_monotonic_decreasing

    @final
    def _memory_usage(self, deep: bool=False) -> int:
        """
        Memory usage of the values.

        Parameters
        ----------
        deep : bool, default False
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption.

        Returns
        -------
        bytes used

        See Also
        --------
        numpy.ndarray.nbytes : Total bytes consumed by the elements of the
            array.

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False or if used on PyPy

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.memory_usage()
        24
        """
        if hasattr(self.array, 'memory_usage'):
            return self.array.memory_usage(deep=deep)
        v = self.array.nbytes
        if deep and is_object_dtype(self.dtype) and (not PYPY):
            values = cast(np.ndarray, self._values)
            v += lib.memory_usage_of_objects(values)
        return v

    @doc(algorithms.factorize, values='', order='', size_hint='', sort=textwrap.dedent('            sort : bool, default False\n                Sort `uniques` and shuffle `codes` to maintain the\n                relationship.\n            '))
    def factorize(self, sort: bool=False, use_na_sentinel: bool=True) -> tuple[npt.NDArray[np.intp], Index]:
        codes, uniques = algorithms.factorize(self._values, sort=sort, use_na_sentinel=use_na_sentinel)
        if uniques.dtype == np.float16:
            uniques = uniques.astype(np.float32)
        if isinstance(self, ABCIndex):
            uniques = self._constructor(uniques)
        else:
            from pandas import Index
            uniques = Index(uniques)
        return (codes, uniques)
    _shared_docs['searchsorted'] = "\n        Find indices where elements should be inserted to maintain order.\n\n        Find the indices into a sorted {klass} `self` such that, if the\n        corresponding elements in `value` were inserted before the indices,\n        the order of `self` would be preserved.\n\n        .. note::\n\n            The {klass} *must* be monotonically sorted, otherwise\n            wrong locations will likely be returned. Pandas does *not*\n            check this for you.\n\n        Parameters\n        ----------\n        value : array-like or scalar\n            Values to insert into `self`.\n        side : {{'left', 'right'}}, optional\n            If 'left', the index of the first suitable location found is given.\n            If 'right', return the last such index.  If there is no suitable\n            index, return either 0 or N (where N is the length of `self`).\n        sorter : 1-D array-like, optional\n            Optional array of integer indices that sort `self` into ascending\n            order. They are typically the result of ``np.argsort``.\n\n        Returns\n        -------\n        int or array of int\n            A scalar or array of insertion points with the\n            same shape as `value`.\n\n        See Also\n        --------\n        sort_values : Sort by the values along either axis.\n        numpy.searchsorted : Similar method from NumPy.\n\n        Notes\n        -----\n        Binary search is used to find the required insertion points.\n\n        Examples\n        --------\n        >>> ser = pd.Series([1, 2, 3])\n        >>> ser\n        0    1\n        1    2\n        2    3\n        dtype: int64\n\n        >>> ser.searchsorted(4)\n        3\n\n        >>> ser.searchsorted([0, 4])\n        array([0, 3])\n\n        >>> ser.searchsorted([1, 3], side='left')\n        array([0, 2])\n\n        >>> ser.searchsorted([1, 3], side='right')\n        array([1, 3])\n\n        >>> ser = pd.Series(pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000']))\n        >>> ser\n        0   2000-03-11\n        1   2000-03-12\n        2   2000-03-13\n        dtype: datetime64[ns]\n\n        >>> ser.searchsorted('3/14/2000')\n        3\n\n        >>> ser = pd.Categorical(\n        ...     ['apple', 'bread', 'bread', 'cheese', 'milk'], ordered=True\n        ... )\n        >>> ser\n        ['apple', 'bread', 'bread', 'cheese', 'milk']\n        Categories (4, object): ['apple' < 'bread' < 'cheese' < 'milk']\n\n        >>> ser.searchsorted('bread')\n        1\n\n        >>> ser.searchsorted(['bread'], side='right')\n        array([3])\n\n        If the values are not monotonically sorted, wrong locations\n        may be returned:\n\n        >>> ser = pd.Series([2, 1, 3])\n        >>> ser\n        0    2\n        1    1\n        2    3\n        dtype: int64\n\n        >>> ser.searchsorted(1)  # doctest: +SKIP\n        0  # wrong result, correct would be 1\n        "

    @overload
    def searchsorted(self, value: ScalarLike_co, side: Literal['left', 'right']=..., sorter: NumpySorter=...) -> np.intp:
        ...

    @overload
    def searchsorted(self, value: npt.ArrayLike | ExtensionArray, side: Literal['left', 'right']=..., sorter: NumpySorter=...) -> npt.NDArray[np.intp]:
        ...

    @doc(_shared_docs['searchsorted'], klass='Index')
    def searchsorted(self, value: NumpyValueArrayLike | ExtensionArray, side: Literal['left', 'right']='left', sorter: NumpySorter | None=None) -> npt.NDArray[np.intp] | np.intp:
        if isinstance(value, ABCDataFrame):
            msg = f'Value must be 1-D array-like or scalar, {type(value).__name__} is not supported'
            raise ValueError(msg)
        values = self._values
        if not isinstance(values, np.ndarray):
            return values.searchsorted(value, side=side, sorter=sorter)
        return algorithms.searchsorted(values, value, side=side, sorter=sorter)

    def drop_duplicates(self, *, keep: DropKeep='first'):
        duplicated = self._duplicated(keep=keep)
        return self[~duplicated]

    @final
    def _duplicated(self, keep: DropKeep='first') -> npt.NDArray[np.bool_]:
        arr = self._values
        if isinstance(arr, ExtensionArray):
            return arr.duplicated(keep=keep)
        return algorithms.duplicated(arr, keep=keep)

    def _arith_method(self, other, op):
        res_name = ops.get_op_result_name(self, other)
        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)
        rvalues = ops.maybe_prepare_scalar_for_op(rvalues, lvalues.shape)
        rvalues = ensure_wrapped_if_datetimelike(rvalues)
        if isinstance(rvalues, range):
            rvalues = np.arange(rvalues.start, rvalues.stop, rvalues.step)
        with np.errstate(all='ignore'):
            result = ops.arithmetic_op(lvalues, rvalues, op)
        return self._construct_result(result, name=res_name)

    def _construct_result(self, result, name):
        """
        Construct an appropriately-wrapped result from the ArrayLike result
        of an arithmetic-like operation.
        """
        raise AbstractMethodError(self)