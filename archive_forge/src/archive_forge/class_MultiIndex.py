from __future__ import annotations
from collections.abc import (
from functools import wraps
from sys import getsizeof
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import coerce_indexer_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_array_like
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import validate_putmask
from pandas.core.arrays import (
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.construction import sanitize_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.frozen import FrozenList
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
from pandas.io.formats.printing import (
class MultiIndex(Index):
    """
    A multi-level, or hierarchical, index object for pandas objects.

    Parameters
    ----------
    levels : sequence of arrays
        The unique labels for each level.
    codes : sequence of arrays
        Integers for each level designating which label at each location.
    sortorder : optional int
        Level of sortedness (must be lexicographically sorted by that
        level).
    names : optional sequence of objects
        Names for each of the index levels. (name is accepted for compat).
    copy : bool, default False
        Copy the meta-data.
    verify_integrity : bool, default True
        Check that the levels/codes are consistent and valid.

    Attributes
    ----------
    names
    levels
    codes
    nlevels
    levshape
    dtypes

    Methods
    -------
    from_arrays
    from_tuples
    from_product
    from_frame
    set_levels
    set_codes
    to_frame
    to_flat_index
    sortlevel
    droplevel
    swaplevel
    reorder_levels
    remove_unused_levels
    get_level_values
    get_indexer
    get_loc
    get_locs
    get_loc_level
    drop

    See Also
    --------
    MultiIndex.from_arrays  : Convert list of arrays to MultiIndex.
    MultiIndex.from_product : Create a MultiIndex from the cartesian product
                              of iterables.
    MultiIndex.from_tuples  : Convert list of tuples to a MultiIndex.
    MultiIndex.from_frame   : Make a MultiIndex from a DataFrame.
    Index : The base pandas Index type.

    Notes
    -----
    See the `user guide
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html>`__
    for more.

    Examples
    --------
    A new ``MultiIndex`` is typically constructed using one of the helper
    methods :meth:`MultiIndex.from_arrays`, :meth:`MultiIndex.from_product`
    and :meth:`MultiIndex.from_tuples`. For example (using ``.from_arrays``):

    >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
    >>> pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
    MultiIndex([(1,  'red'),
                (1, 'blue'),
                (2,  'red'),
                (2, 'blue')],
               names=['number', 'color'])

    See further examples for how to construct a MultiIndex in the doc strings
    of the mentioned helper methods.
    """
    _hidden_attrs = Index._hidden_attrs | frozenset()
    _typ = 'multiindex'
    _names: list[Hashable | None] = []
    _levels = FrozenList()
    _codes = FrozenList()
    _comparables = ['names']
    sortorder: int | None

    def __new__(cls, levels=None, codes=None, sortorder=None, names=None, dtype=None, copy: bool=False, name=None, verify_integrity: bool=True) -> Self:
        if name is not None:
            names = name
        if levels is None or codes is None:
            raise TypeError('Must pass both levels and codes')
        if len(levels) != len(codes):
            raise ValueError('Length of levels and codes must be the same.')
        if len(levels) == 0:
            raise ValueError('Must pass non-zero number of levels/codes')
        result = object.__new__(cls)
        result._cache = {}
        result._set_levels(levels, copy=copy, validate=False)
        result._set_codes(codes, copy=copy, validate=False)
        result._names = [None] * len(levels)
        if names is not None:
            result._set_names(names)
        if sortorder is not None:
            result.sortorder = int(sortorder)
        else:
            result.sortorder = sortorder
        if verify_integrity:
            new_codes = result._verify_integrity()
            result._codes = new_codes
        result._reset_identity()
        result._references = None
        return result

    def _validate_codes(self, level: list, code: list):
        """
        Reassign code values as -1 if their corresponding levels are NaN.

        Parameters
        ----------
        code : list
            Code to reassign.
        level : list
            Level to check for missing values (NaN, NaT, None).

        Returns
        -------
        new code where code value = -1 if it corresponds
        to a level with missing values (NaN, NaT, None).
        """
        null_mask = isna(level)
        if np.any(null_mask):
            code = np.where(null_mask[code], -1, code)
        return code

    def _verify_integrity(self, codes: list | None=None, levels: list | None=None, levels_to_verify: list[int] | range | None=None):
        """
        Parameters
        ----------
        codes : optional list
            Codes to check for validity. Defaults to current codes.
        levels : optional list
            Levels to check for validity. Defaults to current levels.
        levels_to_validate: optional list
            Specifies the levels to verify.

        Raises
        ------
        ValueError
            If length of levels and codes don't match, if the codes for any
            level would exceed level bounds, or there are any duplicate levels.

        Returns
        -------
        new codes where code value = -1 if it corresponds to a
        NaN level.
        """
        codes = codes or self.codes
        levels = levels or self.levels
        if levels_to_verify is None:
            levels_to_verify = range(len(levels))
        if len(levels) != len(codes):
            raise ValueError('Length of levels and codes must match. NOTE: this index is in an inconsistent state.')
        codes_length = len(codes[0])
        for i in levels_to_verify:
            level = levels[i]
            level_codes = codes[i]
            if len(level_codes) != codes_length:
                raise ValueError(f'Unequal code lengths: {[len(code_) for code_ in codes]}')
            if len(level_codes) and level_codes.max() >= len(level):
                raise ValueError(f'On level {i}, code max ({level_codes.max()}) >= length of level ({len(level)}). NOTE: this index is in an inconsistent state')
            if len(level_codes) and level_codes.min() < -1:
                raise ValueError(f'On level {i}, code value ({level_codes.min()}) < -1')
            if not level.is_unique:
                raise ValueError(f'Level values must be unique: {list(level)} on level {i}')
        if self.sortorder is not None:
            if self.sortorder > _lexsort_depth(self.codes, self.nlevels):
                raise ValueError(f'Value for sortorder must be inferior or equal to actual lexsort_depth: sortorder {self.sortorder} with lexsort_depth {_lexsort_depth(self.codes, self.nlevels)}')
        result_codes = []
        for i in range(len(levels)):
            if i in levels_to_verify:
                result_codes.append(self._validate_codes(levels[i], codes[i]))
            else:
                result_codes.append(codes[i])
        new_codes = FrozenList(result_codes)
        return new_codes

    @classmethod
    def from_arrays(cls, arrays, sortorder: int | None=None, names: Sequence[Hashable] | Hashable | lib.NoDefault=lib.no_default) -> MultiIndex:
        """
        Convert arrays to MultiIndex.

        Parameters
        ----------
        arrays : list / sequence of array-likes
            Each array-like gives one level's value for each data point.
            len(arrays) is the number of levels.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        >>> pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        error_msg = 'Input must be a list / sequence of array-likes.'
        if not is_list_like(arrays):
            raise TypeError(error_msg)
        if is_iterator(arrays):
            arrays = list(arrays)
        for array in arrays:
            if not is_list_like(array):
                raise TypeError(error_msg)
        for i in range(1, len(arrays)):
            if len(arrays[i]) != len(arrays[i - 1]):
                raise ValueError('all arrays must be same length')
        codes, levels = factorize_from_iterables(arrays)
        if names is lib.no_default:
            names = [getattr(arr, 'name', None) for arr in arrays]
        return cls(levels=levels, codes=codes, sortorder=sortorder, names=names, verify_integrity=False)

    @classmethod
    @names_compat
    def from_tuples(cls, tuples: Iterable[tuple[Hashable, ...]], sortorder: int | None=None, names: Sequence[Hashable] | Hashable | None=None) -> MultiIndex:
        """
        Convert list of tuples to MultiIndex.

        Parameters
        ----------
        tuples : list / sequence of tuple-likes
            Each tuple is the index of one row/column.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> tuples = [(1, 'red'), (1, 'blue'),
        ...           (2, 'red'), (2, 'blue')]
        >>> pd.MultiIndex.from_tuples(tuples, names=('number', 'color'))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        if not is_list_like(tuples):
            raise TypeError('Input must be a list / sequence of tuple-likes.')
        if is_iterator(tuples):
            tuples = list(tuples)
        tuples = cast(Collection[tuple[Hashable, ...]], tuples)
        if len(tuples) and all((isinstance(e, tuple) and (not e) for e in tuples)):
            codes = [np.zeros(len(tuples))]
            levels = [Index(com.asarray_tuplesafe(tuples, dtype=np.dtype('object')))]
            return cls(levels=levels, codes=codes, sortorder=sortorder, names=names, verify_integrity=False)
        arrays: list[Sequence[Hashable]]
        if len(tuples) == 0:
            if names is None:
                raise TypeError('Cannot infer number of levels from empty list')
            arrays = [[]] * len(names)
        elif isinstance(tuples, (np.ndarray, Index)):
            if isinstance(tuples, Index):
                tuples = np.asarray(tuples._values)
            arrays = list(lib.tuples_to_object_array(tuples).T)
        elif isinstance(tuples, list):
            arrays = list(lib.to_object_array_tuples(tuples).T)
        else:
            arrs = zip(*tuples)
            arrays = cast(list[Sequence[Hashable]], arrs)
        return cls.from_arrays(arrays, sortorder=sortorder, names=names)

    @classmethod
    def from_product(cls, iterables: Sequence[Iterable[Hashable]], sortorder: int | None=None, names: Sequence[Hashable] | Hashable | lib.NoDefault=lib.no_default) -> MultiIndex:
        """
        Make a MultiIndex from the cartesian product of multiple iterables.

        Parameters
        ----------
        iterables : list / sequence of iterables
            Each iterable has unique labels for each level of the index.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.
            If not explicitly provided, names will be inferred from the
            elements of iterables if an element has a name attribute.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> numbers = [0, 1, 2]
        >>> colors = ['green', 'purple']
        >>> pd.MultiIndex.from_product([numbers, colors],
        ...                            names=['number', 'color'])
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        """
        from pandas.core.reshape.util import cartesian_product
        if not is_list_like(iterables):
            raise TypeError('Input must be a list / sequence of iterables.')
        if is_iterator(iterables):
            iterables = list(iterables)
        codes, levels = factorize_from_iterables(iterables)
        if names is lib.no_default:
            names = [getattr(it, 'name', None) for it in iterables]
        codes = cartesian_product(codes)
        return cls(levels, codes, sortorder=sortorder, names=names)

    @classmethod
    def from_frame(cls, df: DataFrame, sortorder: int | None=None, names: Sequence[Hashable] | Hashable | None=None) -> MultiIndex:
        """
        Make a MultiIndex from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted to MultiIndex.
        sortorder : int, optional
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list-like, optional
            If no names are provided, use the column names, or tuple of column
            names if the columns is a MultiIndex. If a sequence, overwrite
            names with the given sequence.

        Returns
        -------
        MultiIndex
            The MultiIndex representation of the given DataFrame.

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.

        Examples
        --------
        >>> df = pd.DataFrame([['HI', 'Temp'], ['HI', 'Precip'],
        ...                    ['NJ', 'Temp'], ['NJ', 'Precip']],
        ...                   columns=['a', 'b'])
        >>> df
              a       b
        0    HI    Temp
        1    HI  Precip
        2    NJ    Temp
        3    NJ  Precip

        >>> pd.MultiIndex.from_frame(df)
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['a', 'b'])

        Using explicit names, instead of the column names

        >>> pd.MultiIndex.from_frame(df, names=['state', 'observation'])
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['state', 'observation'])
        """
        if not isinstance(df, ABCDataFrame):
            raise TypeError('Input must be a DataFrame')
        column_names, columns = zip(*df.items())
        names = column_names if names is None else names
        return cls.from_arrays(columns, sortorder=sortorder, names=names)

    @cache_readonly
    def _values(self) -> np.ndarray:
        values = []
        for i in range(self.nlevels):
            index = self.levels[i]
            codes = self.codes[i]
            vals = index
            if isinstance(vals.dtype, CategoricalDtype):
                vals = cast('CategoricalIndex', vals)
                vals = vals._data._internal_get_values()
            if isinstance(vals.dtype, ExtensionDtype) or lib.is_np_dtype(vals.dtype, 'mM'):
                vals = vals.astype(object)
            vals = np.array(vals, copy=False)
            vals = algos.take_nd(vals, codes, fill_value=index._na_value)
            values.append(vals)
        arr = lib.fast_zip(values)
        return arr

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def array(self):
        """
        Raises a ValueError for `MultiIndex` because there's no single
        array backing a MultiIndex.

        Raises
        ------
        ValueError
        """
        raise ValueError("MultiIndex has no single backing array. Use 'MultiIndex.to_numpy()' to get a NumPy array of tuples.")

    @cache_readonly
    def dtypes(self) -> Series:
        """
        Return the dtypes as a Series for the underlying MultiIndex.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_product([(0, 1, 2), ('green', 'purple')],
        ...                                  names=['number', 'color'])
        >>> idx
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        >>> idx.dtypes
        number     int64
        color     object
        dtype: object
        """
        from pandas import Series
        names = com.fill_missing_names([level.name for level in self.levels])
        return Series([level.dtype for level in self.levels], index=Index(names))

    def __len__(self) -> int:
        return len(self.codes[0])

    @property
    def size(self) -> int:
        """
        Return the number of elements in the underlying data.
        """
        return len(self)

    @cache_readonly
    def levels(self) -> FrozenList:
        """
        Levels of the MultiIndex.

        Levels refer to the different hierarchical levels or layers in a MultiIndex.
        In a MultiIndex, each level represents a distinct dimension or category of
        the index.

        To access the levels, you can use the levels attribute of the MultiIndex,
        which returns a tuple of Index objects. Each Index object represents a
        level in the MultiIndex and contains the unique values found in that
        specific level.

        If a MultiIndex is created with levels A, B, C, and the DataFrame using
        it filters out all rows of the level C, MultiIndex.levels will still
        return A, B, C.

        Examples
        --------
        >>> index = pd.MultiIndex.from_product([['mammal'],
        ...                                     ('goat', 'human', 'cat', 'dog')],
        ...                                    names=['Category', 'Animals'])
        >>> leg_num = pd.DataFrame(data=(4, 2, 4, 4), index=index, columns=['Legs'])
        >>> leg_num
                          Legs
        Category Animals
        mammal   goat        4
                 human       2
                 cat         4
                 dog         4

        >>> leg_num.index.levels
        FrozenList([['mammal'], ['cat', 'dog', 'goat', 'human']])

        MultiIndex levels will not change even if the DataFrame using the MultiIndex
        does not contain all them anymore.
        See how "human" is not in the DataFrame, but it is still in levels:

        >>> large_leg_num = leg_num[leg_num.Legs > 2]
        >>> large_leg_num
                          Legs
        Category Animals
        mammal   goat        4
                 cat         4
                 dog         4

        >>> large_leg_num.index.levels
        FrozenList([['mammal'], ['cat', 'dog', 'goat', 'human']])
        """
        result = [x._rename(name=name) for x, name in zip(self._levels, self._names)]
        for level in result:
            level._no_setting_name = True
        return FrozenList(result)

    def _set_levels(self, levels, *, level=None, copy: bool=False, validate: bool=True, verify_integrity: bool=False) -> None:
        if validate:
            if len(levels) == 0:
                raise ValueError('Must set non-zero number of levels.')
            if level is None and len(levels) != self.nlevels:
                raise ValueError('Length of levels must match number of levels.')
            if level is not None and len(levels) != len(level):
                raise ValueError('Length of levels must match length of level.')
        if level is None:
            new_levels = FrozenList((ensure_index(lev, copy=copy)._view() for lev in levels))
            level_numbers = list(range(len(new_levels)))
        else:
            level_numbers = [self._get_level_number(lev) for lev in level]
            new_levels_list = list(self._levels)
            for lev_num, lev in zip(level_numbers, levels):
                new_levels_list[lev_num] = ensure_index(lev, copy=copy)._view()
            new_levels = FrozenList(new_levels_list)
        if verify_integrity:
            new_codes = self._verify_integrity(levels=new_levels, levels_to_verify=level_numbers)
            self._codes = new_codes
        names = self.names
        self._levels = new_levels
        if any(names):
            self._set_names(names)
        self._reset_cache()

    def set_levels(self, levels, *, level=None, verify_integrity: bool=True) -> MultiIndex:
        """
        Set new levels on MultiIndex. Defaults to returning new index.

        Parameters
        ----------
        levels : sequence or list of sequence
            New level(s) to apply.
        level : int, level name, or sequence of int/level names (default None)
            Level(s) to set (None for all levels).
        verify_integrity : bool, default True
            If True, checks that levels and codes are compatible.

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [
        ...         (1, "one"),
        ...         (1, "two"),
        ...         (2, "one"),
        ...         (2, "two"),
        ...         (3, "one"),
        ...         (3, "two")
        ...     ],
        ...     names=["foo", "bar"]
        ... )
        >>> idx
        MultiIndex([(1, 'one'),
            (1, 'two'),
            (2, 'one'),
            (2, 'two'),
            (3, 'one'),
            (3, 'two')],
           names=['foo', 'bar'])

        >>> idx.set_levels([['a', 'b', 'c'], [1, 2]])
        MultiIndex([('a', 1),
                    ('a', 2),
                    ('b', 1),
                    ('b', 2),
                    ('c', 1),
                    ('c', 2)],
                   names=['foo', 'bar'])
        >>> idx.set_levels(['a', 'b', 'c'], level=0)
        MultiIndex([('a', 'one'),
                    ('a', 'two'),
                    ('b', 'one'),
                    ('b', 'two'),
                    ('c', 'one'),
                    ('c', 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_levels(['a', 'b'], level='bar')
        MultiIndex([(1, 'a'),
                    (1, 'b'),
                    (2, 'a'),
                    (2, 'b'),
                    (3, 'a'),
                    (3, 'b')],
                   names=['foo', 'bar'])

        If any of the levels passed to ``set_levels()`` exceeds the
        existing length, all of the values from that argument will
        be stored in the MultiIndex levels, though the values will
        be truncated in the MultiIndex output.

        >>> idx.set_levels([['a', 'b', 'c'], [1, 2, 3, 4]], level=[0, 1])
        MultiIndex([('a', 1),
            ('a', 2),
            ('b', 1),
            ('b', 2),
            ('c', 1),
            ('c', 2)],
           names=['foo', 'bar'])
        >>> idx.set_levels([['a', 'b', 'c'], [1, 2, 3, 4]], level=[0, 1]).levels
        FrozenList([['a', 'b', 'c'], [1, 2, 3, 4]])
        """
        if isinstance(levels, Index):
            pass
        elif is_array_like(levels):
            levels = Index(levels)
        elif is_list_like(levels):
            levels = list(levels)
        level, levels = _require_listlike(level, levels, 'Levels')
        idx = self._view()
        idx._reset_identity()
        idx._set_levels(levels, level=level, validate=True, verify_integrity=verify_integrity)
        return idx

    @property
    def nlevels(self) -> int:
        """
        Integer number of levels in this MultiIndex.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a'], ['b'], ['c']])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.nlevels
        3
        """
        return len(self._levels)

    @property
    def levshape(self) -> Shape:
        """
        A tuple with the length of each level.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a'], ['b'], ['c']])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.levshape
        (1, 1, 1)
        """
        return tuple((len(x) for x in self.levels))

    @property
    def codes(self) -> FrozenList:
        return self._codes

    def _set_codes(self, codes, *, level=None, copy: bool=False, validate: bool=True, verify_integrity: bool=False) -> None:
        if validate:
            if level is None and len(codes) != self.nlevels:
                raise ValueError('Length of codes must match number of levels')
            if level is not None and len(codes) != len(level):
                raise ValueError('Length of codes must match length of levels.')
        level_numbers: list[int] | range
        if level is None:
            new_codes = FrozenList((_coerce_indexer_frozen(level_codes, lev, copy=copy).view() for lev, level_codes in zip(self._levels, codes)))
            level_numbers = range(len(new_codes))
        else:
            level_numbers = [self._get_level_number(lev) for lev in level]
            new_codes_list = list(self._codes)
            for lev_num, level_codes in zip(level_numbers, codes):
                lev = self.levels[lev_num]
                new_codes_list[lev_num] = _coerce_indexer_frozen(level_codes, lev, copy=copy)
            new_codes = FrozenList(new_codes_list)
        if verify_integrity:
            new_codes = self._verify_integrity(codes=new_codes, levels_to_verify=level_numbers)
        self._codes = new_codes
        self._reset_cache()

    def set_codes(self, codes, *, level=None, verify_integrity: bool=True) -> MultiIndex:
        """
        Set new codes on MultiIndex. Defaults to returning new index.

        Parameters
        ----------
        codes : sequence or list of sequence
            New codes to apply.
        level : int, level name, or sequence of int/level names (default None)
            Level(s) to set (None for all levels).
        verify_integrity : bool, default True
            If True, checks that levels and codes are compatible.

        Returns
        -------
        new index (of same type and class...etc) or None
            The same type as the caller or None if ``inplace=True``.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [(1, "one"), (1, "two"), (2, "one"), (2, "two")], names=["foo", "bar"]
        ... )
        >>> idx
        MultiIndex([(1, 'one'),
            (1, 'two'),
            (2, 'one'),
            (2, 'two')],
           names=['foo', 'bar'])

        >>> idx.set_codes([[1, 0, 1, 0], [0, 0, 1, 1]])
        MultiIndex([(2, 'one'),
                    (1, 'one'),
                    (2, 'two'),
                    (1, 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_codes([1, 0, 1, 0], level=0)
        MultiIndex([(2, 'one'),
                    (1, 'two'),
                    (2, 'one'),
                    (1, 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_codes([0, 0, 1, 1], level='bar')
        MultiIndex([(1, 'one'),
                    (1, 'one'),
                    (2, 'two'),
                    (2, 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_codes([[1, 0, 1, 0], [0, 0, 1, 1]], level=[0, 1])
        MultiIndex([(2, 'one'),
                    (1, 'one'),
                    (2, 'two'),
                    (1, 'two')],
                   names=['foo', 'bar'])
        """
        level, codes = _require_listlike(level, codes, 'Codes')
        idx = self._view()
        idx._reset_identity()
        idx._set_codes(codes, level=level, verify_integrity=verify_integrity)
        return idx

    @cache_readonly
    def _engine(self):
        sizes = np.ceil(np.log2([len(level) + libindex.multiindex_nulls_shift for level in self.levels]))
        lev_bits = np.cumsum(sizes[::-1])[::-1]
        offsets = np.concatenate([lev_bits[1:], [0]]).astype('uint64')
        if lev_bits[0] > 64:
            return MultiIndexPyIntEngine(self.levels, self.codes, offsets)
        return MultiIndexUIntEngine(self.levels, self.codes, offsets)

    @property
    def _constructor(self) -> Callable[..., MultiIndex]:
        return type(self).from_tuples

    @doc(Index._shallow_copy)
    def _shallow_copy(self, values: np.ndarray, name=lib.no_default) -> MultiIndex:
        names = name if name is not lib.no_default else self.names
        return type(self).from_tuples(values, sortorder=None, names=names)

    def _view(self) -> MultiIndex:
        result = type(self)(levels=self.levels, codes=self.codes, sortorder=self.sortorder, names=self.names, verify_integrity=False)
        result._cache = self._cache.copy()
        result._cache.pop('levels', None)
        return result

    def copy(self, names=None, deep: bool=False, name=None) -> Self:
        """
        Make a copy of this object.

        Names, dtype, levels and codes can be passed and will be set on new copy.

        Parameters
        ----------
        names : sequence, optional
        deep : bool, default False
        name : Label
            Kept for compatibility with 1-dimensional Index. Should not be used.

        Returns
        -------
        MultiIndex

        Notes
        -----
        In most cases, there should be no functional difference from using
        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.
        This could be potentially expensive on large MultiIndex objects.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a'], ['b'], ['c']])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.copy()
        MultiIndex([('a', 'b', 'c')],
                   )
        """
        names = self._validate_names(name=name, names=names, deep=deep)
        keep_id = not deep
        levels, codes = (None, None)
        if deep:
            from copy import deepcopy
            levels = deepcopy(self.levels)
            codes = deepcopy(self.codes)
        levels = levels if levels is not None else self.levels
        codes = codes if codes is not None else self.codes
        new_index = type(self)(levels=levels, codes=codes, sortorder=self.sortorder, names=names, verify_integrity=False)
        new_index._cache = self._cache.copy()
        new_index._cache.pop('levels', None)
        if keep_id:
            new_index._id = self._id
        return new_index

    def __array__(self, dtype=None) -> np.ndarray:
        """the array interface, return my values"""
        return self.values

    def view(self, cls=None) -> Self:
        """this is defined as a copy with the same identity"""
        result = self.copy()
        result._id = self._id
        return result

    @doc(Index.__contains__)
    def __contains__(self, key: Any) -> bool:
        hash(key)
        try:
            self.get_loc(key)
            return True
        except (LookupError, TypeError, ValueError):
            return False

    @cache_readonly
    def dtype(self) -> np.dtype:
        return np.dtype('O')

    def _is_memory_usage_qualified(self) -> bool:
        """return a boolean if we need a qualified .info display"""

        def f(level) -> bool:
            return 'mixed' in level or 'string' in level or 'unicode' in level
        return any((f(level) for level in self._inferred_type_levels))

    @doc(Index.memory_usage)
    def memory_usage(self, deep: bool=False) -> int:
        return self._nbytes(deep)

    @cache_readonly
    def nbytes(self) -> int:
        """return the number of bytes in the underlying data"""
        return self._nbytes(False)

    def _nbytes(self, deep: bool=False) -> int:
        """
        return the number of bytes in the underlying data
        deeply introspect the level data if deep=True

        include the engine hashtable

        *this is in internal routine*

        """
        objsize = 24
        level_nbytes = sum((i.memory_usage(deep=deep) for i in self.levels))
        label_nbytes = sum((i.nbytes for i in self.codes))
        names_nbytes = sum((getsizeof(i, objsize) for i in self.names))
        result = level_nbytes + label_nbytes + names_nbytes
        result += self._engine.sizeof(deep=deep)
        return result

    def _formatter_func(self, tup):
        """
        Formats each item in tup according to its level's formatter function.
        """
        formatter_funcs = [level._formatter_func for level in self.levels]
        return tuple((func(val) for func, val in zip(formatter_funcs, tup)))

    def _get_values_for_csv(self, *, na_rep: str='nan', **kwargs) -> npt.NDArray[np.object_]:
        new_levels = []
        new_codes = []
        for level, level_codes in zip(self.levels, self.codes):
            level_strs = level._get_values_for_csv(na_rep=na_rep, **kwargs)
            mask = level_codes == -1
            if mask.any():
                nan_index = len(level_strs)
                level_strs = level_strs.astype(str)
                level_strs = np.append(level_strs, na_rep)
                assert not level_codes.flags.writeable
                level_codes = level_codes.copy()
                level_codes[mask] = nan_index
            new_levels.append(level_strs)
            new_codes.append(level_codes)
        if len(new_levels) == 1:
            return Index(new_levels[0].take(new_codes[0]))._get_values_for_csv()
        else:
            mi = MultiIndex(levels=new_levels, codes=new_codes, names=self.names, sortorder=self.sortorder, verify_integrity=False)
            return mi._values

    def format(self, name: bool | None=None, formatter: Callable | None=None, na_rep: str | None=None, names: bool=False, space: int=2, sparsify=None, adjoin: bool=True) -> list:
        warnings.warn(f'{type(self).__name__}.format is deprecated and will be removed in a future version. Convert using index.astype(str) or index.map(formatter) instead.', FutureWarning, stacklevel=find_stack_level())
        if name is not None:
            names = name
        if len(self) == 0:
            return []
        stringified_levels = []
        for lev, level_codes in zip(self.levels, self.codes):
            na = na_rep if na_rep is not None else _get_na_rep(lev.dtype)
            if len(lev) > 0:
                formatted = lev.take(level_codes).format(formatter=formatter)
                mask = level_codes == -1
                if mask.any():
                    formatted = np.array(formatted, dtype=object)
                    formatted[mask] = na
                    formatted = formatted.tolist()
            else:
                formatted = [pprint_thing(na if isna(x) else x, escape_chars=('\t', '\r', '\n')) for x in algos.take_nd(lev._values, level_codes)]
            stringified_levels.append(formatted)
        result_levels = []
        for lev, lev_name in zip(stringified_levels, self.names):
            level = []
            if names:
                level.append(pprint_thing(lev_name, escape_chars=('\t', '\r', '\n')) if lev_name is not None else '')
            level.extend(np.array(lev, dtype=object))
            result_levels.append(level)
        if sparsify is None:
            sparsify = get_option('display.multi_sparse')
        if sparsify:
            sentinel: Literal[''] | bool | lib.NoDefault = ''
            assert isinstance(sparsify, bool) or sparsify is lib.no_default
            if sparsify in [False, lib.no_default]:
                sentinel = sparsify
            result_levels = sparsify_labels(result_levels, start=int(names), sentinel=sentinel)
        if adjoin:
            adj = get_adjustment()
            return adj.adjoin(space, *result_levels).split('\n')
        else:
            return result_levels

    def _format_multi(self, *, include_names: bool, sparsify: bool | None | lib.NoDefault, formatter: Callable | None=None) -> list:
        if len(self) == 0:
            return []
        stringified_levels = []
        for lev, level_codes in zip(self.levels, self.codes):
            na = _get_na_rep(lev.dtype)
            if len(lev) > 0:
                taken = formatted = lev.take(level_codes)
                formatted = taken._format_flat(include_name=False, formatter=formatter)
                mask = level_codes == -1
                if mask.any():
                    formatted = np.array(formatted, dtype=object)
                    formatted[mask] = na
                    formatted = formatted.tolist()
            else:
                formatted = [pprint_thing(na if isna(x) else x, escape_chars=('\t', '\r', '\n')) for x in algos.take_nd(lev._values, level_codes)]
            stringified_levels.append(formatted)
        result_levels = []
        for lev, lev_name in zip(stringified_levels, self.names):
            level = []
            if include_names:
                level.append(pprint_thing(lev_name, escape_chars=('\t', '\r', '\n')) if lev_name is not None else '')
            level.extend(np.array(lev, dtype=object))
            result_levels.append(level)
        if sparsify is None:
            sparsify = get_option('display.multi_sparse')
        if sparsify:
            sentinel: Literal[''] | bool | lib.NoDefault = ''
            assert isinstance(sparsify, bool) or sparsify is lib.no_default
            if sparsify is lib.no_default:
                sentinel = sparsify
            result_levels = sparsify_labels(result_levels, start=int(include_names), sentinel=sentinel)
        return result_levels

    def _get_names(self) -> FrozenList:
        return FrozenList(self._names)

    def _set_names(self, names, *, level=None, validate: bool=True):
        """
        Set new names on index. Each name has to be a hashable type.

        Parameters
        ----------
        values : str or sequence
            name(s) to set
        level : int, level name, or sequence of int/level names (default None)
            If the index is a MultiIndex (hierarchical), level(s) to set (None
            for all levels).  Otherwise level must be None
        validate : bool, default True
            validate that the names match level lengths

        Raises
        ------
        TypeError if each name is not hashable.

        Notes
        -----
        sets names on levels. WARNING: mutates!

        Note that you generally want to set this *after* changing levels, so
        that it only acts on copies
        """
        if names is not None and (not is_list_like(names)):
            raise ValueError('Names should be list-like for a MultiIndex')
        names = list(names)
        if validate:
            if level is not None and len(names) != len(level):
                raise ValueError('Length of names must match length of level.')
            if level is None and len(names) != self.nlevels:
                raise ValueError('Length of names must match number of levels in MultiIndex.')
        if level is None:
            level = range(self.nlevels)
        else:
            level = [self._get_level_number(lev) for lev in level]
        for lev, name in zip(level, names):
            if name is not None:
                if not is_hashable(name):
                    raise TypeError(f'{type(self).__name__}.name must be a hashable type')
            self._names[lev] = name
        self._reset_cache()
    names = property(fset=_set_names, fget=_get_names, doc="\n        Names of levels in MultiIndex.\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex.from_arrays(\n        ... [[1, 2], [3, 4], [5, 6]], names=['x', 'y', 'z'])\n        >>> mi\n        MultiIndex([(1, 3, 5),\n                    (2, 4, 6)],\n                   names=['x', 'y', 'z'])\n        >>> mi.names\n        FrozenList(['x', 'y', 'z'])\n        ")

    @cache_readonly
    def inferred_type(self) -> str:
        return 'mixed'

    def _get_level_number(self, level) -> int:
        count = self.names.count(level)
        if count > 1 and (not is_integer(level)):
            raise ValueError(f'The name {level} occurs multiple times, use a level number')
        try:
            level = self.names.index(level)
        except ValueError as err:
            if not is_integer(level):
                raise KeyError(f'Level {level} not found') from err
            if level < 0:
                level += self.nlevels
                if level < 0:
                    orig_level = level - self.nlevels
                    raise IndexError(f'Too many levels: Index has only {self.nlevels} levels, {orig_level} is not a valid level number') from err
            elif level >= self.nlevels:
                raise IndexError(f'Too many levels: Index has only {self.nlevels} levels, not {level + 1}') from err
        return level

    @cache_readonly
    def is_monotonic_increasing(self) -> bool:
        """
        Return a boolean if the values are equal or increasing.
        """
        if any((-1 in code for code in self.codes)):
            return False
        if all((level.is_monotonic_increasing for level in self.levels)):
            return libalgos.is_lexsorted([x.astype('int64', copy=False) for x in self.codes])
        values = [self._get_level_values(i)._values for i in reversed(range(len(self.levels)))]
        try:
            sort_order = np.lexsort(values)
            return Index(sort_order).is_monotonic_increasing
        except TypeError:
            return Index(self._values).is_monotonic_increasing

    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        """
        Return a boolean if the values are equal or decreasing.
        """
        return self[::-1].is_monotonic_increasing

    @cache_readonly
    def _inferred_type_levels(self) -> list[str]:
        """return a list of the inferred types, one for each level"""
        return [i.inferred_type for i in self.levels]

    @doc(Index.duplicated)
    def duplicated(self, keep: DropKeep='first') -> npt.NDArray[np.bool_]:
        shape = tuple((len(lev) for lev in self.levels))
        ids = get_group_index(self.codes, shape, sort=False, xnull=False)
        return duplicated(ids, keep)
    _duplicated = duplicated

    def fillna(self, value=None, downcast=None):
        """
        fillna is not implemented for MultiIndex
        """
        raise NotImplementedError('isna is not defined for MultiIndex')

    @doc(Index.dropna)
    def dropna(self, how: AnyAll='any') -> MultiIndex:
        nans = [level_codes == -1 for level_codes in self.codes]
        if how == 'any':
            indexer = np.any(nans, axis=0)
        elif how == 'all':
            indexer = np.all(nans, axis=0)
        else:
            raise ValueError(f'invalid how option: {how}')
        new_codes = [level_codes[~indexer] for level_codes in self.codes]
        return self.set_codes(codes=new_codes)

    def _get_level_values(self, level: int, unique: bool=False) -> Index:
        """
        Return vector of label values for requested level,
        equal to the length of the index

        **this is an internal method**

        Parameters
        ----------
        level : int
        unique : bool, default False
            if True, drop duplicated values

        Returns
        -------
        Index
        """
        lev = self.levels[level]
        level_codes = self.codes[level]
        name = self._names[level]
        if unique:
            level_codes = algos.unique(level_codes)
        filled = algos.take_nd(lev._values, level_codes, fill_value=lev._na_value)
        return lev._shallow_copy(filled, name=name)

    def get_level_values(self, level) -> Index:
        """
        Return vector of label values for requested level.

        Length of returned vector is equal to the length of the index.

        Parameters
        ----------
        level : int or str
            ``level`` is either the integer position of the level in the
            MultiIndex, or the name of the level.

        Returns
        -------
        Index
            Values is a level of this MultiIndex converted to
            a single :class:`Index` (or subclass thereof).

        Notes
        -----
        If the level contains missing values, the result may be casted to
        ``float`` with missing values specified as ``NaN``. This is because
        the level is converted to a regular ``Index``.

        Examples
        --------
        Create a MultiIndex:

        >>> mi = pd.MultiIndex.from_arrays((list('abc'), list('def')))
        >>> mi.names = ['level_1', 'level_2']

        Get level values by supplying level as either integer or name:

        >>> mi.get_level_values(0)
        Index(['a', 'b', 'c'], dtype='object', name='level_1')
        >>> mi.get_level_values('level_2')
        Index(['d', 'e', 'f'], dtype='object', name='level_2')

        If a level contains missing values, the return type of the level
        may be cast to ``float``.

        >>> pd.MultiIndex.from_arrays([[1, None, 2], [3, 4, 5]]).dtypes
        level_0    int64
        level_1    int64
        dtype: object
        >>> pd.MultiIndex.from_arrays([[1, None, 2], [3, 4, 5]]).get_level_values(0)
        Index([1.0, nan, 2.0], dtype='float64')
        """
        level = self._get_level_number(level)
        values = self._get_level_values(level)
        return values

    @doc(Index.unique)
    def unique(self, level=None):
        if level is None:
            return self.drop_duplicates()
        else:
            level = self._get_level_number(level)
            return self._get_level_values(level=level, unique=True)

    def to_frame(self, index: bool=True, name=lib.no_default, allow_duplicates: bool=False) -> DataFrame:
        """
        Create a DataFrame with the levels of the MultiIndex as columns.

        Column ordering is determined by the DataFrame constructor with data as
        a dict.

        Parameters
        ----------
        index : bool, default True
            Set the index of the returned DataFrame as the original MultiIndex.

        name : list / sequence of str, optional
            The passed names should substitute index level names.

        allow_duplicates : bool, optional default False
            Allow duplicate column labels to be created.

            .. versionadded:: 1.5.0

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame : Two-dimensional, size-mutable, potentially heterogeneous
            tabular data.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a', 'b'], ['c', 'd']])
        >>> mi
        MultiIndex([('a', 'c'),
                    ('b', 'd')],
                   )

        >>> df = mi.to_frame()
        >>> df
             0  1
        a c  a  c
        b d  b  d

        >>> df = mi.to_frame(index=False)
        >>> df
           0  1
        0  a  c
        1  b  d

        >>> df = mi.to_frame(name=['x', 'y'])
        >>> df
             x  y
        a c  a  c
        b d  b  d
        """
        from pandas import DataFrame
        if name is not lib.no_default:
            if not is_list_like(name):
                raise TypeError("'name' must be a list / sequence of column names.")
            if len(name) != len(self.levels):
                raise ValueError("'name' should have same length as number of levels on index.")
            idx_names = name
        else:
            idx_names = self._get_level_names()
        if not allow_duplicates and len(set(idx_names)) != len(idx_names):
            raise ValueError('Cannot create duplicate column labels if allow_duplicates is False')
        result = DataFrame({level: self._get_level_values(level) for level in range(len(self.levels))}, copy=False)
        result.columns = idx_names
        if index:
            result.index = self
        return result

    def to_flat_index(self) -> Index:
        """
        Convert a MultiIndex to an Index of Tuples containing the level values.

        Returns
        -------
        pd.Index
            Index with the MultiIndex data represented in Tuples.

        See Also
        --------
        MultiIndex.from_tuples : Convert flat index back to MultiIndex.

        Notes
        -----
        This method will simply return the caller if called by anything other
        than a MultiIndex.

        Examples
        --------
        >>> index = pd.MultiIndex.from_product(
        ...     [['foo', 'bar'], ['baz', 'qux']],
        ...     names=['a', 'b'])
        >>> index.to_flat_index()
        Index([('foo', 'baz'), ('foo', 'qux'),
               ('bar', 'baz'), ('bar', 'qux')],
              dtype='object')
        """
        return Index(self._values, tupleize_cols=False)

    def _is_lexsorted(self) -> bool:
        """
        Return True if the codes are lexicographically sorted.

        Returns
        -------
        bool

        Examples
        --------
        In the below examples, the first level of the MultiIndex is sorted because
        a<b<c, so there is no need to look at the next level.

        >>> pd.MultiIndex.from_arrays([['a', 'b', 'c'],
        ...                            ['d', 'e', 'f']])._is_lexsorted()
        True
        >>> pd.MultiIndex.from_arrays([['a', 'b', 'c'],
        ...                            ['d', 'f', 'e']])._is_lexsorted()
        True

        In case there is a tie, the lexicographical sorting looks
        at the next level of the MultiIndex.

        >>> pd.MultiIndex.from_arrays([[0, 1, 1], ['a', 'b', 'c']])._is_lexsorted()
        True
        >>> pd.MultiIndex.from_arrays([[0, 1, 1], ['a', 'c', 'b']])._is_lexsorted()
        False
        >>> pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'],
        ...                            ['aa', 'bb', 'aa', 'bb']])._is_lexsorted()
        True
        >>> pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'],
        ...                            ['bb', 'aa', 'aa', 'bb']])._is_lexsorted()
        False
        """
        return self._lexsort_depth == self.nlevels

    @cache_readonly
    def _lexsort_depth(self) -> int:
        """
        Compute and return the lexsort_depth, the number of levels of the
        MultiIndex that are sorted lexically

        Returns
        -------
        int
        """
        if self.sortorder is not None:
            return self.sortorder
        return _lexsort_depth(self.codes, self.nlevels)

    def _sort_levels_monotonic(self, raise_if_incomparable: bool=False) -> MultiIndex:
        """
        This is an *internal* function.

        Create a new MultiIndex from the current to monotonically sorted
        items IN the levels. This does not actually make the entire MultiIndex
        monotonic, JUST the levels.

        The resulting MultiIndex will have the same outward
        appearance, meaning the same .values and ordering. It will also
        be .equals() to the original.

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],
        ...                    codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        >>> mi
        MultiIndex([('a', 'bb'),
                    ('a', 'aa'),
                    ('b', 'bb'),
                    ('b', 'aa')],
                   )

        >>> mi.sort_values()
        MultiIndex([('a', 'aa'),
                    ('a', 'bb'),
                    ('b', 'aa'),
                    ('b', 'bb')],
                   )
        """
        if self._is_lexsorted() and self.is_monotonic_increasing:
            return self
        new_levels = []
        new_codes = []
        for lev, level_codes in zip(self.levels, self.codes):
            if not lev.is_monotonic_increasing:
                try:
                    indexer = lev.argsort()
                except TypeError:
                    if raise_if_incomparable:
                        raise
                else:
                    lev = lev.take(indexer)
                    indexer = ensure_platform_int(indexer)
                    ri = lib.get_reverse_indexer(indexer, len(indexer))
                    level_codes = algos.take_nd(ri, level_codes, fill_value=-1)
            new_levels.append(lev)
            new_codes.append(level_codes)
        return MultiIndex(new_levels, new_codes, names=self.names, sortorder=self.sortorder, verify_integrity=False)

    def remove_unused_levels(self) -> MultiIndex:
        """
        Create new MultiIndex from current that removes unused levels.

        Unused level(s) means levels that are not expressed in the
        labels. The resulting MultiIndex will have the same outward
        appearance, meaning the same .values and ordering. It will
        also be .equals() to the original.

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex.from_product([range(2), list('ab')])
        >>> mi
        MultiIndex([(0, 'a'),
                    (0, 'b'),
                    (1, 'a'),
                    (1, 'b')],
                   )

        >>> mi[2:]
        MultiIndex([(1, 'a'),
                    (1, 'b')],
                   )

        The 0 from the first level is not represented
        and can be removed

        >>> mi2 = mi[2:].remove_unused_levels()
        >>> mi2.levels
        FrozenList([[1], ['a', 'b']])
        """
        new_levels = []
        new_codes = []
        changed = False
        for lev, level_codes in zip(self.levels, self.codes):
            uniques = np.where(np.bincount(level_codes + 1) > 0)[0] - 1
            has_na = int(len(uniques) and uniques[0] == -1)
            if len(uniques) != len(lev) + has_na:
                if lev.isna().any() and len(uniques) == len(lev):
                    break
                changed = True
                uniques = algos.unique(level_codes)
                if has_na:
                    na_idx = np.where(uniques == -1)[0]
                    uniques[[0, na_idx[0]]] = uniques[[na_idx[0], 0]]
                code_mapping = np.zeros(len(lev) + has_na)
                code_mapping[uniques] = np.arange(len(uniques)) - has_na
                level_codes = code_mapping[level_codes]
                lev = lev.take(uniques[has_na:])
            new_levels.append(lev)
            new_codes.append(level_codes)
        result = self.view()
        if changed:
            result._reset_identity()
            result._set_levels(new_levels, validate=False)
            result._set_codes(new_codes, validate=False)
        return result

    def __reduce__(self):
        """Necessary for making this object picklable"""
        d = {'levels': list(self.levels), 'codes': list(self.codes), 'sortorder': self.sortorder, 'names': list(self.names)}
        return (ibase._new_Index, (type(self), d), None)

    def __getitem__(self, key):
        if is_scalar(key):
            key = com.cast_scalar_indexer(key)
            retval = []
            for lev, level_codes in zip(self.levels, self.codes):
                if level_codes[key] == -1:
                    retval.append(np.nan)
                else:
                    retval.append(lev[level_codes[key]])
            return tuple(retval)
        else:
            sortorder = None
            if com.is_bool_indexer(key):
                key = np.asarray(key, dtype=bool)
                sortorder = self.sortorder
            elif isinstance(key, slice):
                if key.step is None or key.step > 0:
                    sortorder = self.sortorder
            elif isinstance(key, Index):
                key = np.asarray(key)
            new_codes = [level_codes[key] for level_codes in self.codes]
            return MultiIndex(levels=self.levels, codes=new_codes, names=self.names, sortorder=sortorder, verify_integrity=False)

    def _getitem_slice(self: MultiIndex, slobj: slice) -> MultiIndex:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        sortorder = None
        if slobj.step is None or slobj.step > 0:
            sortorder = self.sortorder
        new_codes = [level_codes[slobj] for level_codes in self.codes]
        return type(self)(levels=self.levels, codes=new_codes, names=self._names, sortorder=sortorder, verify_integrity=False)

    @Appender(_index_shared_docs['take'] % _index_doc_kwargs)
    def take(self: MultiIndex, indices, axis: Axis=0, allow_fill: bool=True, fill_value=None, **kwargs) -> MultiIndex:
        nv.validate_take((), kwargs)
        indices = ensure_platform_int(indices)
        allow_fill = self._maybe_disallow_fill(allow_fill, fill_value, indices)
        na_value = -1
        taken = [lab.take(indices) for lab in self.codes]
        if allow_fill:
            mask = indices == -1
            if mask.any():
                masked = []
                for new_label in taken:
                    label_values = new_label
                    label_values[mask] = na_value
                    masked.append(np.asarray(label_values))
                taken = masked
        return MultiIndex(levels=self.levels, codes=taken, names=self.names, verify_integrity=False)

    def append(self, other):
        """
        Append a collection of Index options together.

        Parameters
        ----------
        other : Index or list/tuple of indices

        Returns
        -------
        Index
            The combined index.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a'], ['b']])
        >>> mi
        MultiIndex([('a', 'b')],
                   )
        >>> mi.append(mi)
        MultiIndex([('a', 'b'), ('a', 'b')],
                   )
        """
        if not isinstance(other, (list, tuple)):
            other = [other]
        if all((isinstance(o, MultiIndex) and o.nlevels >= self.nlevels for o in other)):
            codes = []
            levels = []
            names = []
            for i in range(self.nlevels):
                level_values = self.levels[i]
                for mi in other:
                    level_values = level_values.union(mi.levels[i])
                level_codes = [recode_for_categories(mi.codes[i], mi.levels[i], level_values, copy=False) for mi in [self, *other]]
                level_name = self.names[i]
                if any((mi.names[i] != level_name for mi in other)):
                    level_name = None
                codes.append(np.concatenate(level_codes))
                levels.append(level_values)
                names.append(level_name)
            return MultiIndex(codes=codes, levels=levels, names=names, verify_integrity=False)
        to_concat = (self._values,) + tuple((k._values for k in other))
        new_tuples = np.concatenate(to_concat)
        try:
            return MultiIndex.from_tuples(new_tuples)
        except (TypeError, IndexError):
            return Index(new_tuples)

    def argsort(self, *args, na_position: str='last', **kwargs) -> npt.NDArray[np.intp]:
        target = self._sort_levels_monotonic(raise_if_incomparable=True)
        keys = [lev.codes for lev in target._get_codes_for_sorting()]
        return lexsort_indexer(keys, na_position=na_position, codes_given=True)

    @Appender(_index_shared_docs['repeat'] % _index_doc_kwargs)
    def repeat(self, repeats: int, axis=None) -> MultiIndex:
        nv.validate_repeat((), {'axis': axis})
        repeats = ensure_platform_int(repeats)
        return MultiIndex(levels=self.levels, codes=[level_codes.view(np.ndarray).astype(np.intp, copy=False).repeat(repeats) for level_codes in self.codes], names=self.names, sortorder=self.sortorder, verify_integrity=False)

    def drop(self, codes, level: Index | np.ndarray | Iterable[Hashable] | None=None, errors: IgnoreRaise='raise') -> MultiIndex:
        """
        Make a new :class:`pandas.MultiIndex` with the passed list of codes deleted.

        Parameters
        ----------
        codes : array-like
            Must be a list of tuples when ``level`` is not specified.
        level : int or level name, default None
        errors : str, default 'raise'

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> idx = pd.MultiIndex.from_product([(0, 1, 2), ('green', 'purple')],
        ...                                  names=["number", "color"])
        >>> idx
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        >>> idx.drop([(1, 'green'), (2, 'purple')])
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1, 'purple'),
                    (2,  'green')],
                   names=['number', 'color'])

        We can also drop from a specific level.

        >>> idx.drop('green', level='color')
        MultiIndex([(0, 'purple'),
                    (1, 'purple'),
                    (2, 'purple')],
                   names=['number', 'color'])

        >>> idx.drop([1, 2], level=0)
        MultiIndex([(0,  'green'),
                    (0, 'purple')],
                   names=['number', 'color'])
        """
        if level is not None:
            return self._drop_from_level(codes, level, errors)
        if not isinstance(codes, (np.ndarray, Index)):
            try:
                codes = com.index_labels_to_array(codes, dtype=np.dtype('object'))
            except ValueError:
                pass
        inds = []
        for level_codes in codes:
            try:
                loc = self.get_loc(level_codes)
                if isinstance(loc, int):
                    inds.append(loc)
                elif isinstance(loc, slice):
                    step = loc.step if loc.step is not None else 1
                    inds.extend(range(loc.start, loc.stop, step))
                elif com.is_bool_indexer(loc):
                    if self._lexsort_depth == 0:
                        warnings.warn('dropping on a non-lexsorted multi-index without a level parameter may impact performance.', PerformanceWarning, stacklevel=find_stack_level())
                    loc = loc.nonzero()[0]
                    inds.extend(loc)
                else:
                    msg = f'unsupported indexer of type {type(loc)}'
                    raise AssertionError(msg)
            except KeyError:
                if errors != 'ignore':
                    raise
        return self.delete(inds)

    def _drop_from_level(self, codes, level, errors: IgnoreRaise='raise') -> MultiIndex:
        codes = com.index_labels_to_array(codes)
        i = self._get_level_number(level)
        index = self.levels[i]
        values = index.get_indexer(codes)
        nan_codes = isna(codes)
        values[np.equal(nan_codes, False) & (values == -1)] = -2
        if index.shape[0] == self.shape[0]:
            values[np.equal(nan_codes, True)] = -2
        not_found = codes[values == -2]
        if len(not_found) != 0 and errors != 'ignore':
            raise KeyError(f'labels {not_found} not found in level')
        mask = ~algos.isin(self.codes[i], values)
        return self[mask]

    def swaplevel(self, i=-2, j=-1) -> MultiIndex:
        """
        Swap level i with level j.

        Calling this method does not change the ordering of the values.

        Parameters
        ----------
        i : int, str, default -2
            First level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.
        j : int, str, default -1
            Second level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.

        Returns
        -------
        MultiIndex
            A new MultiIndex.

        See Also
        --------
        Series.swaplevel : Swap levels i and j in a MultiIndex.
        DataFrame.swaplevel : Swap levels i and j in a MultiIndex on a
            particular axis.

        Examples
        --------
        >>> mi = pd.MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],
        ...                    codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        >>> mi
        MultiIndex([('a', 'bb'),
                    ('a', 'aa'),
                    ('b', 'bb'),
                    ('b', 'aa')],
                   )
        >>> mi.swaplevel(0, 1)
        MultiIndex([('bb', 'a'),
                    ('aa', 'a'),
                    ('bb', 'b'),
                    ('aa', 'b')],
                   )
        """
        new_levels = list(self.levels)
        new_codes = list(self.codes)
        new_names = list(self.names)
        i = self._get_level_number(i)
        j = self._get_level_number(j)
        new_levels[i], new_levels[j] = (new_levels[j], new_levels[i])
        new_codes[i], new_codes[j] = (new_codes[j], new_codes[i])
        new_names[i], new_names[j] = (new_names[j], new_names[i])
        return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)

    def reorder_levels(self, order) -> MultiIndex:
        """
        Rearrange levels using input order. May not drop or duplicate levels.

        Parameters
        ----------
        order : list of int or list of str
            List representing new level order. Reference level by number
            (position) or by key (label).

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([[1, 2], [3, 4]], names=['x', 'y'])
        >>> mi
        MultiIndex([(1, 3),
                    (2, 4)],
                   names=['x', 'y'])

        >>> mi.reorder_levels(order=[1, 0])
        MultiIndex([(3, 1),
                    (4, 2)],
                   names=['y', 'x'])

        >>> mi.reorder_levels(order=['y', 'x'])
        MultiIndex([(3, 1),
                    (4, 2)],
                   names=['y', 'x'])
        """
        order = [self._get_level_number(i) for i in order]
        result = self._reorder_ilevels(order)
        return result

    def _reorder_ilevels(self, order) -> MultiIndex:
        if len(order) != self.nlevels:
            raise AssertionError(f'Length of order must be same as number of levels ({self.nlevels}), got {len(order)}')
        new_levels = [self.levels[i] for i in order]
        new_codes = [self.codes[i] for i in order]
        new_names = [self.names[i] for i in order]
        return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)

    def _recode_for_new_levels(self, new_levels, copy: bool=True) -> Generator[np.ndarray, None, None]:
        if len(new_levels) > self.nlevels:
            raise AssertionError(f'Length of new_levels ({len(new_levels)}) must be <= self.nlevels ({self.nlevels})')
        for i in range(len(new_levels)):
            yield recode_for_categories(self.codes[i], self.levels[i], new_levels[i], copy=copy)

    def _get_codes_for_sorting(self) -> list[Categorical]:
        """
        we are categorizing our codes by using the
        available categories (all, not just observed)
        excluding any missing ones (-1); this is in preparation
        for sorting, where we need to disambiguate that -1 is not
        a valid valid
        """

        def cats(level_codes):
            return np.arange(np.array(level_codes).max() + 1 if len(level_codes) else 0, dtype=level_codes.dtype)
        return [Categorical.from_codes(level_codes, cats(level_codes), True, validate=False) for level_codes in self.codes]

    def sortlevel(self, level: IndexLabel=0, ascending: bool | list[bool]=True, sort_remaining: bool=True, na_position: str='first') -> tuple[MultiIndex, npt.NDArray[np.intp]]:
        """
        Sort MultiIndex at the requested level.

        The result will respect the original ordering of the associated
        factor at that level.

        Parameters
        ----------
        level : list-like, int or str, default 0
            If a string is given, must be a name of the level.
            If list-like must be names or ints of levels.
        ascending : bool, default True
            False to sort in descending order.
            Can also be a list to specify a directed ordering.
        sort_remaining : sort by the remaining levels after level
        na_position : {'first' or 'last'}, default 'first'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.

            .. versionadded:: 2.1.0

        Returns
        -------
        sorted_index : pd.MultiIndex
            Resulting index.
        indexer : np.ndarray[np.intp]
            Indices of output values in original index.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([[0, 0], [2, 1]])
        >>> mi
        MultiIndex([(0, 2),
                    (0, 1)],
                   )

        >>> mi.sortlevel()
        (MultiIndex([(0, 1),
                    (0, 2)],
                   ), array([1, 0]))

        >>> mi.sortlevel(sort_remaining=False)
        (MultiIndex([(0, 2),
                    (0, 1)],
                   ), array([0, 1]))

        >>> mi.sortlevel(1)
        (MultiIndex([(0, 1),
                    (0, 2)],
                   ), array([1, 0]))

        >>> mi.sortlevel(1, ascending=False)
        (MultiIndex([(0, 2),
                    (0, 1)],
                   ), array([0, 1]))
        """
        if not is_list_like(level):
            level = [level]
        level = [self._get_level_number(lev) for lev in level]
        sortorder = None
        codes = [self.codes[lev] for lev in level]
        if isinstance(ascending, list):
            if not len(level) == len(ascending):
                raise ValueError('level must have same length as ascending')
        elif sort_remaining:
            codes.extend([self.codes[lev] for lev in range(len(self.levels)) if lev not in level])
        else:
            sortorder = level[0]
        indexer = lexsort_indexer(codes, orders=ascending, na_position=na_position, codes_given=True)
        indexer = ensure_platform_int(indexer)
        new_codes = [level_codes.take(indexer) for level_codes in self.codes]
        new_index = MultiIndex(codes=new_codes, levels=self.levels, names=self.names, sortorder=sortorder, verify_integrity=False)
        return (new_index, indexer)

    def _wrap_reindex_result(self, target, indexer, preserve_names: bool):
        if not isinstance(target, MultiIndex):
            if indexer is None:
                target = self
            elif (indexer >= 0).all():
                target = self.take(indexer)
            else:
                try:
                    target = MultiIndex.from_tuples(target)
                except TypeError:
                    return target
        target = self._maybe_preserve_names(target, preserve_names)
        return target

    def _maybe_preserve_names(self, target: Index, preserve_names: bool) -> Index:
        if preserve_names and target.nlevels == self.nlevels and (target.names != self.names):
            target = target.copy(deep=False)
            target.names = self.names
        return target

    def _check_indexing_error(self, key) -> None:
        if not is_hashable(key) or is_iterator(key):
            raise InvalidIndexError(key)

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        """
        Should integer key(s) be treated as positional?
        """
        return self.levels[0]._should_fallback_to_positional

    def _get_indexer_strict(self, key, axis_name: str) -> tuple[Index, npt.NDArray[np.intp]]:
        keyarr = key
        if not isinstance(keyarr, Index):
            keyarr = com.asarray_tuplesafe(keyarr)
        if len(keyarr) and (not isinstance(keyarr[0], tuple)):
            indexer = self._get_indexer_level_0(keyarr)
            self._raise_if_missing(key, indexer, axis_name)
            return (self[indexer], indexer)
        return super()._get_indexer_strict(key, axis_name)

    def _raise_if_missing(self, key, indexer, axis_name: str) -> None:
        keyarr = key
        if not isinstance(key, Index):
            keyarr = com.asarray_tuplesafe(key)
        if len(keyarr) and (not isinstance(keyarr[0], tuple)):
            mask = indexer == -1
            if mask.any():
                check = self.levels[0].get_indexer(keyarr)
                cmask = check == -1
                if cmask.any():
                    raise KeyError(f'{keyarr[cmask]} not in index')
                raise KeyError(f'{keyarr} not in index')
        else:
            return super()._raise_if_missing(key, indexer, axis_name)

    def _get_indexer_level_0(self, target) -> npt.NDArray[np.intp]:
        """
        Optimized equivalent to `self.get_level_values(0).get_indexer_for(target)`.
        """
        lev = self.levels[0]
        codes = self._codes[0]
        cat = Categorical.from_codes(codes=codes, categories=lev, validate=False)
        ci = Index(cat)
        return ci.get_indexer_for(target)

    def get_slice_bound(self, label: Hashable | Sequence[Hashable], side: Literal['left', 'right']) -> int:
        """
        For an ordered MultiIndex, compute slice bound
        that corresponds to given label.

        Returns leftmost (one-past-the-rightmost if `side=='right') position
        of given label.

        Parameters
        ----------
        label : object or tuple of objects
        side : {'left', 'right'}

        Returns
        -------
        int
            Index of label.

        Notes
        -----
        This method only works if level 0 index of the MultiIndex is lexsorted.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abbc'), list('gefd')])

        Get the locations from the leftmost 'b' in the first level
        until the end of the multiindex:

        >>> mi.get_slice_bound('b', side="left")
        1

        Like above, but if you get the locations from the rightmost
        'b' in the first level and 'f' in the second level:

        >>> mi.get_slice_bound(('b','f'), side="right")
        3

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.
        """
        if not isinstance(label, tuple):
            label = (label,)
        return self._partial_tup_index(label, side=side)

    def slice_locs(self, start=None, end=None, step=None) -> tuple[int, int]:
        """
        For an ordered MultiIndex, compute the slice locations for input
        labels.

        The input labels can be tuples representing partial levels, e.g. for a
        MultiIndex with 3 levels, you can pass a single value (corresponding to
        the first level), or a 1-, 2-, or 3-tuple.

        Parameters
        ----------
        start : label or tuple, default None
            If None, defaults to the beginning
        end : label or tuple
            If None, defaults to the end
        step : int or None
            Slice step

        Returns
        -------
        (start, end) : (int, int)

        Notes
        -----
        This method only works if the MultiIndex is properly lexsorted. So,
        if only the first 2 levels of a 3-level MultiIndex are lexsorted,
        you can only pass two levels to ``.slice_locs``.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abbd'), list('deff')],
        ...                                names=['A', 'B'])

        Get the slice locations from the beginning of 'b' in the first level
        until the end of the multiindex:

        >>> mi.slice_locs(start='b')
        (1, 4)

        Like above, but stop at the end of 'b' in the first level and 'f' in
        the second level:

        >>> mi.slice_locs(start='b', end=('b', 'f'))
        (1, 3)

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.
        """
        return super().slice_locs(start, end, step)

    def _partial_tup_index(self, tup: tuple, side: Literal['left', 'right']='left'):
        if len(tup) > self._lexsort_depth:
            raise UnsortedIndexError(f'Key length ({len(tup)}) was greater than MultiIndex lexsort depth ({self._lexsort_depth})')
        n = len(tup)
        start, end = (0, len(self))
        zipped = zip(tup, self.levels, self.codes)
        for k, (lab, lev, level_codes) in enumerate(zipped):
            section = level_codes[start:end]
            loc: npt.NDArray[np.intp] | np.intp | int
            if lab not in lev and (not isna(lab)):
                try:
                    loc = algos.searchsorted(lev, lab, side=side)
                except TypeError as err:
                    raise TypeError(f'Level type mismatch: {lab}') from err
                if not is_integer(loc):
                    raise TypeError(f'Level type mismatch: {lab}')
                if side == 'right' and loc >= 0:
                    loc -= 1
                return start + algos.searchsorted(section, loc, side=side)
            idx = self._get_loc_single_level_index(lev, lab)
            if isinstance(idx, slice) and k < n - 1:
                start = idx.start
                end = idx.stop
            elif k < n - 1:
                end = start + algos.searchsorted(section, idx, side='right')
                start = start + algos.searchsorted(section, idx, side='left')
            elif isinstance(idx, slice):
                idx = idx.start
                return start + algos.searchsorted(section, idx, side=side)
            else:
                return start + algos.searchsorted(section, idx, side=side)

    def _get_loc_single_level_index(self, level_index: Index, key: Hashable) -> int:
        """
        If key is NA value, location of index unify as -1.

        Parameters
        ----------
        level_index: Index
        key : label

        Returns
        -------
        loc : int
            If key is NA value, loc is -1
            Else, location of key in index.

        See Also
        --------
        Index.get_loc : The get_loc method for (single-level) index.
        """
        if is_scalar(key) and isna(key):
            return -1
        else:
            return level_index.get_loc(key)

    def get_loc(self, key):
        """
        Get location for a label or a tuple of labels.

        The location is returned as an integer/slice or boolean
        mask.

        Parameters
        ----------
        key : label or tuple of labels (one for each level)

        Returns
        -------
        int, slice object or boolean mask
            If the key is past the lexsort depth, the return may be a
            boolean mask array, otherwise it is always a slice or int.

        See Also
        --------
        Index.get_loc : The get_loc method for (single-level) index.
        MultiIndex.slice_locs : Get slice location given start label(s) and
                                end label(s).
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.

        Notes
        -----
        The key cannot be a slice, list of same-level labels, a boolean mask,
        or a sequence of such. If you want to use those, use
        :meth:`MultiIndex.get_locs` instead.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')])

        >>> mi.get_loc('b')
        slice(1, 3, None)

        >>> mi.get_loc(('b', 'e'))
        1
        """
        self._check_indexing_error(key)

        def _maybe_to_slice(loc):
            """convert integer indexer to boolean mask or slice if possible"""
            if not isinstance(loc, np.ndarray) or loc.dtype != np.intp:
                return loc
            loc = lib.maybe_indices_to_slice(loc, len(self))
            if isinstance(loc, slice):
                return loc
            mask = np.empty(len(self), dtype='bool')
            mask.fill(False)
            mask[loc] = True
            return mask
        if not isinstance(key, tuple):
            loc = self._get_level_indexer(key, level=0)
            return _maybe_to_slice(loc)
        keylen = len(key)
        if self.nlevels < keylen:
            raise KeyError(f'Key length ({keylen}) exceeds index depth ({self.nlevels})')
        if keylen == self.nlevels and self.is_unique:
            try:
                return self._engine.get_loc(key)
            except KeyError as err:
                raise KeyError(key) from err
            except TypeError:
                loc, _ = self.get_loc_level(key, list(range(self.nlevels)))
                return loc
        i = self._lexsort_depth
        lead_key, follow_key = (key[:i], key[i:])
        if not lead_key:
            start = 0
            stop = len(self)
        else:
            try:
                start, stop = self.slice_locs(lead_key, lead_key)
            except TypeError as err:
                raise KeyError(key) from err
        if start == stop:
            raise KeyError(key)
        if not follow_key:
            return slice(start, stop)
        warnings.warn('indexing past lexsort depth may impact performance.', PerformanceWarning, stacklevel=find_stack_level())
        loc = np.arange(start, stop, dtype=np.intp)
        for i, k in enumerate(follow_key, len(lead_key)):
            mask = self.codes[i][loc] == self._get_loc_single_level_index(self.levels[i], k)
            if not mask.all():
                loc = loc[mask]
            if not len(loc):
                raise KeyError(key)
        return _maybe_to_slice(loc) if len(loc) != stop - start else slice(start, stop)

    def get_loc_level(self, key, level: IndexLabel=0, drop_level: bool=True):
        """
        Get location and sliced index for requested label(s)/level(s).

        Parameters
        ----------
        key : label or sequence of labels
        level : int/level name or list thereof, optional
        drop_level : bool, default True
            If ``False``, the resulting index will not drop any level.

        Returns
        -------
        tuple
            A 2-tuple where the elements :

            Element 0: int, slice object or boolean array.

            Element 1: The resulting sliced multiindex/index. If the key
            contains all levels, this will be ``None``.

        See Also
        --------
        MultiIndex.get_loc  : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')],
        ...                                names=['A', 'B'])

        >>> mi.get_loc_level('b')
        (slice(1, 3, None), Index(['e', 'f'], dtype='object', name='B'))

        >>> mi.get_loc_level('e', level='B')
        (array([False,  True, False]), Index(['b'], dtype='object', name='A'))

        >>> mi.get_loc_level(['b', 'e'])
        (1, None)
        """
        if not isinstance(level, (list, tuple)):
            level = self._get_level_number(level)
        else:
            level = [self._get_level_number(lev) for lev in level]
        loc, mi = self._get_loc_level(key, level=level)
        if not drop_level:
            if lib.is_integer(loc):
                mi = self[loc:loc + 1]
            else:
                mi = self[loc]
        return (loc, mi)

    def _get_loc_level(self, key, level: int | list[int]=0):
        """
        get_loc_level but with `level` known to be positional, not name-based.
        """

        def maybe_mi_droplevels(indexer, levels):
            """
            If level does not exist or all levels were dropped, the exception
            has to be handled outside.
            """
            new_index = self[indexer]
            for i in sorted(levels, reverse=True):
                new_index = new_index._drop_level_numbers([i])
            return new_index
        if isinstance(level, (tuple, list)):
            if len(key) != len(level):
                raise AssertionError('Key for location must have same length as number of levels')
            result = None
            for lev, k in zip(level, key):
                loc, new_index = self._get_loc_level(k, level=lev)
                if isinstance(loc, slice):
                    mask = np.zeros(len(self), dtype=bool)
                    mask[loc] = True
                    loc = mask
                result = loc if result is None else result & loc
            try:
                mi = maybe_mi_droplevels(result, level)
            except ValueError:
                mi = self[result]
            return (result, mi)
        if isinstance(key, list):
            key = tuple(key)
        if isinstance(key, tuple) and level == 0:
            try:
                if key in self.levels[0]:
                    indexer = self._get_level_indexer(key, level=level)
                    new_index = maybe_mi_droplevels(indexer, [0])
                    return (indexer, new_index)
            except (TypeError, InvalidIndexError):
                pass
            if not any((isinstance(k, slice) for k in key)):
                if len(key) == self.nlevels and self.is_unique:
                    try:
                        return (self._engine.get_loc(key), None)
                    except KeyError as err:
                        raise KeyError(key) from err
                    except TypeError:
                        pass
                indexer = self.get_loc(key)
                ilevels = [i for i in range(len(key)) if key[i] != slice(None, None)]
                if len(ilevels) == self.nlevels:
                    if is_integer(indexer):
                        return (indexer, None)
                    ilevels = [i for i in range(len(key)) if (not isinstance(key[i], str) or not self.levels[i]._supports_partial_string_indexing) and key[i] != slice(None, None)]
                    if len(ilevels) == self.nlevels:
                        ilevels = []
                return (indexer, maybe_mi_droplevels(indexer, ilevels))
            else:
                indexer = None
                for i, k in enumerate(key):
                    if not isinstance(k, slice):
                        loc_level = self._get_level_indexer(k, level=i)
                        if isinstance(loc_level, slice):
                            if com.is_null_slice(loc_level) or com.is_full_slice(loc_level, len(self)):
                                continue
                            k_index = np.zeros(len(self), dtype=bool)
                            k_index[loc_level] = True
                        else:
                            k_index = loc_level
                    elif com.is_null_slice(k):
                        continue
                    else:
                        raise TypeError(f'Expected label or tuple of labels, got {key}')
                    if indexer is None:
                        indexer = k_index
                    else:
                        indexer &= k_index
                if indexer is None:
                    indexer = slice(None, None)
                ilevels = [i for i in range(len(key)) if key[i] != slice(None, None)]
                return (indexer, maybe_mi_droplevels(indexer, ilevels))
        else:
            indexer = self._get_level_indexer(key, level=level)
            if isinstance(key, str) and self.levels[level]._supports_partial_string_indexing:
                check = self.levels[level].get_loc(key)
                if not is_integer(check):
                    return (indexer, self[indexer])
            try:
                result_index = maybe_mi_droplevels(indexer, [level])
            except ValueError:
                result_index = self[indexer]
            return (indexer, result_index)

    def _get_level_indexer(self, key, level: int=0, indexer: npt.NDArray[np.bool_] | None=None):
        level_index = self.levels[level]
        level_codes = self.codes[level]

        def convert_indexer(start, stop, step, indexer=indexer, codes=level_codes):
            if indexer is not None:
                codes = codes[indexer]
            if step is None or step == 1:
                new_indexer = (codes >= start) & (codes < stop)
            else:
                r = np.arange(start, stop, step, dtype=codes.dtype)
                new_indexer = algos.isin(codes, r)
            if indexer is None:
                return new_indexer
            indexer = indexer.copy()
            indexer[indexer] = new_indexer
            return indexer
        if isinstance(key, slice):
            step = key.step
            is_negative_step = step is not None and step < 0
            try:
                if key.start is not None:
                    start = level_index.get_loc(key.start)
                elif is_negative_step:
                    start = len(level_index) - 1
                else:
                    start = 0
                if key.stop is not None:
                    stop = level_index.get_loc(key.stop)
                elif is_negative_step:
                    stop = 0
                elif isinstance(start, slice):
                    stop = len(level_index)
                else:
                    stop = len(level_index) - 1
            except KeyError:
                start = stop = level_index.slice_indexer(key.start, key.stop, key.step)
                step = start.step
            if isinstance(start, slice) or isinstance(stop, slice):
                start = getattr(start, 'start', start)
                stop = getattr(stop, 'stop', stop)
                return convert_indexer(start, stop, step)
            elif level > 0 or self._lexsort_depth == 0 or step is not None:
                stop = stop - 1 if is_negative_step else stop + 1
                return convert_indexer(start, stop, step)
            else:
                i = algos.searchsorted(level_codes, start, side='left')
                j = algos.searchsorted(level_codes, stop, side='right')
                return slice(i, j, step)
        else:
            idx = self._get_loc_single_level_index(level_index, key)
            if level > 0 or self._lexsort_depth == 0:
                if isinstance(idx, slice):
                    locs = (level_codes >= idx.start) & (level_codes < idx.stop)
                    return locs
                locs = np.array(level_codes == idx, dtype=bool, copy=False)
                if not locs.any():
                    raise KeyError(key)
                return locs
            if isinstance(idx, slice):
                start = algos.searchsorted(level_codes, idx.start, side='left')
                end = algos.searchsorted(level_codes, idx.stop, side='left')
            else:
                start = algos.searchsorted(level_codes, idx, side='left')
                end = algos.searchsorted(level_codes, idx, side='right')
            if start == end:
                raise KeyError(key)
            return slice(start, end)

    def get_locs(self, seq) -> npt.NDArray[np.intp]:
        """
        Get location for a sequence of labels.

        Parameters
        ----------
        seq : label, slice, list, mask or a sequence of such
           You should use one of the above for each level.
           If a level should not be used, set it to ``slice(None)``.

        Returns
        -------
        numpy.ndarray
            NumPy array of integers suitable for passing to iloc.

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.slice_locs : Get slice location given start label(s) and
                                end label(s).

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')])

        >>> mi.get_locs('b')  # doctest: +SKIP
        array([1, 2], dtype=int64)

        >>> mi.get_locs([slice(None), ['e', 'f']])  # doctest: +SKIP
        array([1, 2], dtype=int64)

        >>> mi.get_locs([[True, False, True], slice('e', 'f')])  # doctest: +SKIP
        array([2], dtype=int64)
        """
        true_slices = [i for i, s in enumerate(com.is_true_slices(seq)) if s]
        if true_slices and true_slices[-1] >= self._lexsort_depth:
            raise UnsortedIndexError(f'MultiIndex slicing requires the index to be lexsorted: slicing on levels {true_slices}, lexsort depth {self._lexsort_depth}')
        if any((x is Ellipsis for x in seq)):
            raise NotImplementedError('MultiIndex does not support indexing with Ellipsis')
        n = len(self)

        def _to_bool_indexer(indexer) -> npt.NDArray[np.bool_]:
            if isinstance(indexer, slice):
                new_indexer = np.zeros(n, dtype=np.bool_)
                new_indexer[indexer] = True
                return new_indexer
            return indexer
        indexer: npt.NDArray[np.bool_] | None = None
        for i, k in enumerate(seq):
            lvl_indexer: npt.NDArray[np.bool_] | slice | None = None
            if com.is_bool_indexer(k):
                if len(k) != n:
                    raise ValueError('cannot index with a boolean indexer that is not the same length as the index')
                lvl_indexer = np.asarray(k)
                if indexer is None:
                    lvl_indexer = lvl_indexer.copy()
            elif is_list_like(k):
                try:
                    lvl_indexer = self._get_level_indexer(k, level=i, indexer=indexer)
                except (InvalidIndexError, TypeError, KeyError) as err:
                    for x in k:
                        if not is_hashable(x):
                            raise err
                        item_indexer = self._get_level_indexer(x, level=i, indexer=indexer)
                        if lvl_indexer is None:
                            lvl_indexer = _to_bool_indexer(item_indexer)
                        elif isinstance(item_indexer, slice):
                            lvl_indexer[item_indexer] = True
                        else:
                            lvl_indexer |= item_indexer
                if lvl_indexer is None:
                    return np.array([], dtype=np.intp)
            elif com.is_null_slice(k):
                if indexer is None and i == len(seq) - 1:
                    return np.arange(n, dtype=np.intp)
                continue
            else:
                lvl_indexer = self._get_level_indexer(k, level=i, indexer=indexer)
            lvl_indexer = _to_bool_indexer(lvl_indexer)
            if indexer is None:
                indexer = lvl_indexer
            else:
                indexer &= lvl_indexer
                if not np.any(indexer) and np.any(lvl_indexer):
                    raise KeyError(seq)
        if indexer is None:
            return np.array([], dtype=np.intp)
        pos_indexer = indexer.nonzero()[0]
        return self._reorder_indexer(seq, pos_indexer)

    def _reorder_indexer(self, seq: tuple[Scalar | Iterable | AnyArrayLike, ...], indexer: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        """
        Reorder an indexer of a MultiIndex (self) so that the labels are in the
        same order as given in seq

        Parameters
        ----------
        seq : label/slice/list/mask or a sequence of such
        indexer: a position indexer of self

        Returns
        -------
        indexer : a sorted position indexer of self ordered as seq
        """
        need_sort = False
        for i, k in enumerate(seq):
            if com.is_null_slice(k) or com.is_bool_indexer(k) or is_scalar(k):
                pass
            elif is_list_like(k):
                if len(k) <= 1:
                    pass
                elif self._is_lexsorted():
                    k_codes = self.levels[i].get_indexer(k)
                    k_codes = k_codes[k_codes >= 0]
                    need_sort = (k_codes[:-1] > k_codes[1:]).any()
                else:
                    need_sort = True
            elif isinstance(k, slice):
                if self._is_lexsorted():
                    need_sort = k.step is not None and k.step < 0
                else:
                    need_sort = True
            else:
                need_sort = True
            if need_sort:
                break
        if not need_sort:
            return indexer
        n = len(self)
        keys: tuple[np.ndarray, ...] = ()
        for i, k in enumerate(seq):
            if is_scalar(k):
                k = [k]
            if com.is_bool_indexer(k):
                new_order = np.arange(n)[indexer]
            elif is_list_like(k):
                if not isinstance(k, (np.ndarray, ExtensionArray, Index, ABCSeries)):
                    k = sanitize_array(k, None)
                k = algos.unique(k)
                key_order_map = np.ones(len(self.levels[i]), dtype=np.uint64) * len(self.levels[i])
                level_indexer = self.levels[i].get_indexer(k)
                level_indexer = level_indexer[level_indexer >= 0]
                key_order_map[level_indexer] = np.arange(len(level_indexer))
                new_order = key_order_map[self.codes[i][indexer]]
            elif isinstance(k, slice) and k.step is not None and (k.step < 0):
                new_order = np.arange(n)[::-1][indexer]
            elif isinstance(k, slice) and k.start is None and (k.stop is None):
                new_order = np.ones((n,), dtype=np.intp)[indexer]
            else:
                new_order = np.arange(n)[indexer]
            keys = (new_order,) + keys
        ind = np.lexsort(keys)
        return indexer[ind]

    def truncate(self, before=None, after=None) -> MultiIndex:
        """
        Slice index between two labels / tuples, return new MultiIndex.

        Parameters
        ----------
        before : label or tuple, can be partial. Default None
            None defaults to start.
        after : label or tuple, can be partial. Default None
            None defaults to end.

        Returns
        -------
        MultiIndex
            The truncated MultiIndex.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a', 'b', 'c'], ['x', 'y', 'z']])
        >>> mi
        MultiIndex([('a', 'x'), ('b', 'y'), ('c', 'z')],
                   )
        >>> mi.truncate(before='a', after='b')
        MultiIndex([('a', 'x'), ('b', 'y')],
                   )
        """
        if after and before and (after < before):
            raise ValueError('after < before')
        i, j = self.levels[0].slice_locs(before, after)
        left, right = self.slice_locs(before, after)
        new_levels = list(self.levels)
        new_levels[0] = new_levels[0][i:j]
        new_codes = [level_codes[left:right] for level_codes in self.codes]
        new_codes[0] = new_codes[0] - i
        return MultiIndex(levels=new_levels, codes=new_codes, names=self._names, verify_integrity=False)

    def equals(self, other: object) -> bool:
        """
        Determines if two MultiIndex objects have the same labeling information
        (the levels themselves do not necessarily have to be the same)

        See Also
        --------
        equal_levels
        """
        if self.is_(other):
            return True
        if not isinstance(other, Index):
            return False
        if len(self) != len(other):
            return False
        if not isinstance(other, MultiIndex):
            if not self._should_compare(other):
                return False
            return array_equivalent(self._values, other._values)
        if self.nlevels != other.nlevels:
            return False
        for i in range(self.nlevels):
            self_codes = self.codes[i]
            other_codes = other.codes[i]
            self_mask = self_codes == -1
            other_mask = other_codes == -1
            if not np.array_equal(self_mask, other_mask):
                return False
            self_codes = self_codes[~self_mask]
            self_values = self.levels[i]._values.take(self_codes)
            other_codes = other_codes[~other_mask]
            other_values = other.levels[i]._values.take(other_codes)
            if len(self_values) == 0 and len(other_values) == 0:
                continue
            if not isinstance(self_values, np.ndarray):
                if not self_values.equals(other_values):
                    return False
            elif not isinstance(other_values, np.ndarray):
                if not other_values.equals(self_values):
                    return False
            elif not array_equivalent(self_values, other_values):
                return False
        return True

    def equal_levels(self, other: MultiIndex) -> bool:
        """
        Return True if the levels of both MultiIndex objects are the same

        """
        if self.nlevels != other.nlevels:
            return False
        for i in range(self.nlevels):
            if not self.levels[i].equals(other.levels[i]):
                return False
        return True

    def _union(self, other, sort) -> MultiIndex:
        other, result_names = self._convert_can_do_setop(other)
        if other.has_duplicates:
            result = super()._union(other, sort)
            if isinstance(result, MultiIndex):
                return result
            return MultiIndex.from_arrays(zip(*result), sortorder=None, names=result_names)
        else:
            right_missing = other.difference(self, sort=False)
            if len(right_missing):
                result = self.append(right_missing)
            else:
                result = self._get_reconciled_name_object(other)
            if sort is not False:
                try:
                    result = result.sort_values()
                except TypeError:
                    if sort is True:
                        raise
                    warnings.warn('The values in the array are unorderable. Pass `sort=False` to suppress this warning.', RuntimeWarning, stacklevel=find_stack_level())
            return result

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        return is_object_dtype(dtype)

    def _get_reconciled_name_object(self, other) -> MultiIndex:
        """
        If the result of a set operation will be self,
        return self, unless the names change, in which
        case make a shallow copy of self.
        """
        names = self._maybe_match_names(other)
        if self.names != names:
            return self.rename(names)
        return self

    def _maybe_match_names(self, other):
        """
        Try to find common names to attach to the result of an operation between
        a and b. Return a consensus list of names if they match at least partly
        or list of None if they have completely different names.
        """
        if len(self.names) != len(other.names):
            return [None] * len(self.names)
        names = []
        for a_name, b_name in zip(self.names, other.names):
            if a_name == b_name:
                names.append(a_name)
            else:
                names.append(None)
        return names

    def _wrap_intersection_result(self, other, result) -> MultiIndex:
        _, result_names = self._convert_can_do_setop(other)
        return result.set_names(result_names)

    def _wrap_difference_result(self, other, result: MultiIndex) -> MultiIndex:
        _, result_names = self._convert_can_do_setop(other)
        if len(result) == 0:
            return result.remove_unused_levels().set_names(result_names)
        else:
            return result.set_names(result_names)

    def _convert_can_do_setop(self, other):
        result_names = self.names
        if not isinstance(other, Index):
            if len(other) == 0:
                return (self[:0], self.names)
            else:
                msg = 'other must be a MultiIndex or a list of tuples'
                try:
                    other = MultiIndex.from_tuples(other, names=self.names)
                except (ValueError, TypeError) as err:
                    raise TypeError(msg) from err
        else:
            result_names = get_unanimous_names(self, other)
        return (other, result_names)

    @doc(Index.astype)
    def astype(self, dtype, copy: bool=True):
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, CategoricalDtype):
            msg = '> 1 ndim Categorical are not supported at this time'
            raise NotImplementedError(msg)
        if not is_object_dtype(dtype):
            raise TypeError('Setting a MultiIndex dtype to anything other than object is not supported')
        if copy is True:
            return self._view()
        return self

    def _validate_fill_value(self, item):
        if isinstance(item, MultiIndex):
            if item.nlevels != self.nlevels:
                raise ValueError('Item must have length equal to number of levels.')
            return item._values
        elif not isinstance(item, tuple):
            item = (item,) + ('',) * (self.nlevels - 1)
        elif len(item) != self.nlevels:
            raise ValueError('Item must have length equal to number of levels.')
        return item

    def putmask(self, mask, value: MultiIndex) -> MultiIndex:
        """
        Return a new MultiIndex of the values set with the mask.

        Parameters
        ----------
        mask : array like
        value : MultiIndex
            Must either be the same length as self or length one

        Returns
        -------
        MultiIndex
        """
        mask, noop = validate_putmask(self, mask)
        if noop:
            return self.copy()
        if len(mask) == len(value):
            subset = value[mask].remove_unused_levels()
        else:
            subset = value.remove_unused_levels()
        new_levels = []
        new_codes = []
        for i, (value_level, level, level_codes) in enumerate(zip(subset.levels, self.levels, self.codes)):
            new_level = level.union(value_level, sort=False)
            value_codes = new_level.get_indexer_for(subset.get_level_values(i))
            new_code = ensure_int64(level_codes)
            new_code[mask] = value_codes
            new_levels.append(new_level)
            new_codes.append(new_code)
        return MultiIndex(levels=new_levels, codes=new_codes, names=self.names, verify_integrity=False)

    def insert(self, loc: int, item) -> MultiIndex:
        """
        Make new MultiIndex inserting new item at location

        Parameters
        ----------
        loc : int
        item : tuple
            Must be same length as number of levels in the MultiIndex

        Returns
        -------
        new_index : Index
        """
        item = self._validate_fill_value(item)
        new_levels = []
        new_codes = []
        for k, level, level_codes in zip(item, self.levels, self.codes):
            if k not in level:
                lev_loc = len(level)
                level = level.insert(lev_loc, k)
            else:
                lev_loc = level.get_loc(k)
            new_levels.append(level)
            new_codes.append(np.insert(ensure_int64(level_codes), loc, lev_loc))
        return MultiIndex(levels=new_levels, codes=new_codes, names=self.names, verify_integrity=False)

    def delete(self, loc) -> MultiIndex:
        """
        Make new index with passed location deleted

        Returns
        -------
        new_index : MultiIndex
        """
        new_codes = [np.delete(level_codes, loc) for level_codes in self.codes]
        return MultiIndex(levels=self.levels, codes=new_codes, names=self.names, verify_integrity=False)

    @doc(Index.isin)
    def isin(self, values, level=None) -> npt.NDArray[np.bool_]:
        if isinstance(values, Generator):
            values = list(values)
        if level is None:
            if len(values) == 0:
                return np.zeros((len(self),), dtype=np.bool_)
            if not isinstance(values, MultiIndex):
                values = MultiIndex.from_tuples(values)
            return values.unique().get_indexer_for(self) != -1
        else:
            num = self._get_level_number(level)
            levs = self.get_level_values(num)
            if levs.size == 0:
                return np.zeros(len(levs), dtype=np.bool_)
            return levs.isin(values)
    rename = Index.set_names
    __add__ = make_invalid_op('__add__')
    __radd__ = make_invalid_op('__radd__')
    __iadd__ = make_invalid_op('__iadd__')
    __sub__ = make_invalid_op('__sub__')
    __rsub__ = make_invalid_op('__rsub__')
    __isub__ = make_invalid_op('__isub__')
    __pow__ = make_invalid_op('__pow__')
    __rpow__ = make_invalid_op('__rpow__')
    __mul__ = make_invalid_op('__mul__')
    __rmul__ = make_invalid_op('__rmul__')
    __floordiv__ = make_invalid_op('__floordiv__')
    __rfloordiv__ = make_invalid_op('__rfloordiv__')
    __truediv__ = make_invalid_op('__truediv__')
    __rtruediv__ = make_invalid_op('__rtruediv__')
    __mod__ = make_invalid_op('__mod__')
    __rmod__ = make_invalid_op('__rmod__')
    __divmod__ = make_invalid_op('__divmod__')
    __rdivmod__ = make_invalid_op('__rdivmod__')
    __neg__ = make_invalid_op('__neg__')
    __pos__ = make_invalid_op('__pos__')
    __abs__ = make_invalid_op('__abs__')
    __invert__ = make_invalid_op('__invert__')