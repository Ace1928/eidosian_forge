from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
class DtypesDescriptor:
    """
    Describes partial dtypes.

    Parameters
    ----------
    known_dtypes : dict[IndexLabel, np.dtype] or pandas.Series, optional
        Columns that we know dtypes for.
    cols_with_unknown_dtypes : list[IndexLabel], optional
        Column names that have unknown dtypes. If specified together with `remaining_dtype`, must describe all
        columns with unknown dtypes, otherwise, the missing columns will be assigned to `remaining_dtype`.
        If `cols_with_unknown_dtypes` is incomplete, you must specify `know_all_names=False`.
    remaining_dtype : np.dtype, optional
        Dtype for columns that are not present neither in `known_dtypes` nor in `cols_with_unknown_dtypes`.
        This parameter is intended to describe columns that we known dtypes for, but don't know their
        names yet. Note, that this parameter DOESN'T describe dtypes for columns from `cols_with_unknown_dtypes`.
    parent_df : PandasDataframe, optional
        Dataframe object for which we describe dtypes. This dataframe will be used to compute
        missing dtypes on ``.materialize()``.
    columns_order : dict[int, IndexLabel], optional
        Order of columns in the dataframe. If specified, must describe all the columns of the dataframe.
    know_all_names : bool, default: True
        Whether `known_dtypes` and `cols_with_unknown_dtypes` contain all column names for this dataframe besides those,
        that are being described by `remaining_dtype`.
        One can't pass `know_all_names=False` together with `remaining_dtype` as this creates ambiguity
        on how to interpret missing columns (whether they belong to `remaining_dtype` or not).
    _schema_is_known : bool, optional
        Whether `known_dtypes` describe all columns in the dataframe. This parameter intended mostly
        for internal use.
    """

    def __init__(self, known_dtypes: Optional[Union[dict[IndexLabel, np.dtype], pandas.Series]]=None, cols_with_unknown_dtypes: Optional[list[IndexLabel]]=None, remaining_dtype: Optional[np.dtype]=None, parent_df: Optional['PandasDataframe']=None, columns_order: Optional[dict[int, IndexLabel]]=None, know_all_names: bool=True, _schema_is_known: Optional[bool]=None):
        if not know_all_names and remaining_dtype is not None:
            raise ValueError("It's not allowed to pass 'remaining_dtype' and 'know_all_names=False' at the same time.")
        self._known_dtypes: dict[IndexLabel, np.dtype] = {} if known_dtypes is None else dict(known_dtypes)
        if known_dtypes is not None and len(self._known_dtypes) != len(known_dtypes):
            raise NotImplementedError('Duplicated column names are not yet supported by DtypesDescriptor')
        if cols_with_unknown_dtypes is not None and len(set(cols_with_unknown_dtypes)) != len(cols_with_unknown_dtypes):
            raise NotImplementedError('Duplicated column names are not yet supported by DtypesDescriptor')
        self._cols_with_unknown_dtypes: list[IndexLabel] = [] if cols_with_unknown_dtypes is None else cols_with_unknown_dtypes
        self._schema_is_known: Optional[bool] = _schema_is_known
        if self._schema_is_known is None:
            self._schema_is_known = False
            if cols_with_unknown_dtypes is not None and know_all_names and (remaining_dtype is None) and (len(self._known_dtypes) > 0):
                self._schema_is_known = len(cols_with_unknown_dtypes) == 0
        self._know_all_names: bool = know_all_names
        self._remaining_dtype: Optional[np.dtype] = remaining_dtype
        self._parent_df: Optional['PandasDataframe'] = parent_df
        if columns_order is None:
            self._columns_order: Optional[dict[int, IndexLabel]] = None
            self.columns_order
        else:
            if remaining_dtype is not None:
                raise ValueError("Passing 'columns_order' and 'remaining_dtype' is ambiguous. You have to manually " + "complete 'known_dtypes' using the information from 'columns_order' and 'remaining_dtype'.")
            elif not self._know_all_names:
                raise ValueError("Passing 'columns_order' and 'know_all_names=False' is ambiguous. You have to manually " + "complete 'cols_with_unknown_dtypes' using the information from 'columns_order' " + "and pass 'know_all_names=True'.")
            elif len(columns_order) != len(self._cols_with_unknown_dtypes) + len(self._known_dtypes):
                raise ValueError("The length of 'columns_order' doesn't match to 'known_dtypes' and 'cols_with_unknown_dtypes'")
            self._columns_order: Optional[dict[int, IndexLabel]] = columns_order

    def update_parent(self, new_parent: 'PandasDataframe'):
        """
        Set new parent dataframe.

        Parameters
        ----------
        new_parent : PandasDataframe
        """
        self._parent_df = new_parent
        LazyProxyCategoricalDtype.update_dtypes(self._known_dtypes, new_parent)
        self.columns_order

    @property
    def columns_order(self) -> Optional[dict[int, IndexLabel]]:
        """
        Get order of columns for the described dataframe if available.

        Returns
        -------
        dict[int, IndexLabel] or None
        """
        if self._columns_order is not None:
            return self._columns_order
        if self._parent_df is None or not self._parent_df.has_materialized_columns:
            return None
        actual_columns = self._parent_df.columns
        self._normalize_self_levels(actual_columns)
        self._columns_order = {i: col for i, col in enumerate(actual_columns)}
        if len(self._columns_order) > len(self._known_dtypes) + len(self._cols_with_unknown_dtypes):
            new_cols = [col for col in self._columns_order.values() if col not in self._known_dtypes and col not in self._cols_with_unknown_dtypes]
            if self._remaining_dtype is not None:
                self._known_dtypes.update({col: self._remaining_dtype for col in new_cols})
                self._remaining_dtype = None
                if len(self._cols_with_unknown_dtypes) == 0:
                    self._schema_is_known = True
            else:
                self._cols_with_unknown_dtypes.extend(new_cols)
        self._know_all_names = True
        return self._columns_order

    def __repr__(self):
        return f'DtypesDescriptor:\n\tknown dtypes: {self._known_dtypes};\n\t' + f'remaining dtype: {self._remaining_dtype};\n\t' + f'cols with unknown dtypes: {self._cols_with_unknown_dtypes};\n\t' + f'schema is known: {self._schema_is_known};\n\t' + f'has parent df: {self._parent_df is not None};\n\t' + f'columns order: {self._columns_order};\n\t' + f'know all names: {self._know_all_names}'

    def __str__(self):
        return self.__repr__()

    def lazy_get(self, ids: list[Union[IndexLabel, int]], numeric_index: bool=False) -> 'DtypesDescriptor':
        """
        Get dtypes descriptor for a subset of columns without triggering any computations.

        Parameters
        ----------
        ids : list of index labels or positional indexers
            Columns for the subset.
        numeric_index : bool, default: False
            Whether `ids` are positional indixes or column labels to take.

        Returns
        -------
        DtypesDescriptor
            Descriptor that describes dtypes for columns specified in `ids`.
        """
        if len(set(ids)) != len(ids):
            raise NotImplementedError('Duplicated column names are not yet supported by DtypesDescriptor')
        if numeric_index:
            if self.columns_order is not None:
                ids = [self.columns_order[i] for i in ids]
            else:
                raise ValueError("Can't lazily get columns by positional indixers if the columns order is unknown")
        result = {}
        unknown_cols = []
        columns_order = {}
        for i, col in enumerate(ids):
            columns_order[i] = col
            if col in self._cols_with_unknown_dtypes:
                unknown_cols.append(col)
                continue
            dtype = self._known_dtypes.get(col)
            if dtype is None and self._remaining_dtype is None:
                unknown_cols.append(col)
            elif dtype is None and self._remaining_dtype is not None:
                result[col] = self._remaining_dtype
            else:
                result[col] = dtype
        remaining_dtype = self._remaining_dtype if len(unknown_cols) != 0 else None
        return DtypesDescriptor(result, unknown_cols, remaining_dtype, self._parent_df, columns_order=columns_order)

    def copy(self) -> 'DtypesDescriptor':
        """
        Get a copy of this descriptor.

        Returns
        -------
        DtypesDescriptor
        """
        return type(self)(columns_order=None if self.columns_order is None else self.columns_order.copy(), known_dtypes=self._known_dtypes.copy(), cols_with_unknown_dtypes=self._cols_with_unknown_dtypes.copy(), remaining_dtype=self._remaining_dtype, parent_df=self._parent_df, know_all_names=self._know_all_names, _schema_is_known=self._schema_is_known)

    def set_index(self, new_index: Union[pandas.Index, 'ModinIndex']) -> 'DtypesDescriptor':
        """
        Set new column names for this descriptor.

        Parameters
        ----------
        new_index : pandas.Index or ModinIndex

        Returns
        -------
        DtypesDescriptor
            New descriptor with updated column names.

        Notes
        -----
        Calling this method on a descriptor that returns ``None`` for ``.columns_order``
        will result into information lose.
        """
        if len(new_index) != len(set(new_index)):
            raise NotImplementedError('Duplicated column names are not yet supported by DtypesDescriptor')
        if self.columns_order is None:
            return DtypesDescriptor(cols_with_unknown_dtypes=new_index, columns_order={i: col for i, col in enumerate(new_index)}, parent_df=self._parent_df, know_all_names=True)
        new_self = self.copy()
        renamer = {old_c: new_index[i] for i, old_c in new_self.columns_order.items()}
        new_self._known_dtypes = {renamer[old_col]: value for old_col, value in new_self._known_dtypes.items()}
        new_self._cols_with_unknown_dtypes = [renamer[old_col] for old_col in new_self._cols_with_unknown_dtypes]
        new_self._columns_order = {i: renamer[old_col] for i, old_col in new_self._columns_order.items()}
        return new_self

    def equals(self, other: 'DtypesDescriptor') -> bool:
        """
        Compare two descriptors for equality.

        Parameters
        ----------
        other : DtypesDescriptor

        Returns
        -------
        bool
        """
        return self._known_dtypes == other._known_dtypes and set(self._cols_with_unknown_dtypes) == set(other._cols_with_unknown_dtypes) and (self._remaining_dtype == other._remaining_dtype) and (self._schema_is_known == other._schema_is_known) and (self.columns_order == other.columns_order) and (self._know_all_names == other._know_all_names)

    @property
    def is_materialized(self) -> bool:
        """
        Whether this descriptor contains information about all dtypes in the dataframe.

        Returns
        -------
        bool
        """
        return self._schema_is_known

    def _materialize_all_names(self):
        """Materialize missing column names."""
        if self._know_all_names:
            return
        all_cols = self._parent_df.columns
        self._normalize_self_levels(all_cols)
        for col in all_cols:
            if col not in self._known_dtypes and col not in self._cols_with_unknown_dtypes:
                self._cols_with_unknown_dtypes.append(col)
        self._know_all_names = True

    def _materialize_cols_with_unknown_dtypes(self):
        """Compute dtypes for cols specified in `._cols_with_unknown_dtypes`."""
        if len(self._known_dtypes) == 0 and len(self._cols_with_unknown_dtypes) == 0 and (not self._know_all_names):
            subset = None
        else:
            if not self._know_all_names:
                self._materialize_all_names()
            subset = self._cols_with_unknown_dtypes
        if subset is None or len(subset) > 0:
            self._known_dtypes.update(self._parent_df._compute_dtypes(subset))
        self._know_all_names = True
        self._cols_with_unknown_dtypes = []

    def materialize(self):
        """Complete information about dtypes."""
        if self.is_materialized:
            return
        if self._parent_df is None:
            raise RuntimeError("It's not allowed to call '.materialize()' before '._parent_df' is specified.")
        self._materialize_cols_with_unknown_dtypes()
        if self._remaining_dtype is not None:
            cols = self._parent_df.columns
            self._normalize_self_levels(cols)
            self._known_dtypes.update({col: self._remaining_dtype for col in cols if col not in self._known_dtypes})
        if self.columns_order is not None:
            assert len(self.columns_order) == len(self._known_dtypes)
            self._known_dtypes = {self.columns_order[i]: self._known_dtypes[self.columns_order[i]] for i in range(len(self.columns_order))}
        self._schema_is_known = True
        self._remaining_dtype = None
        self._parent_df = None

    def to_series(self) -> pandas.Series:
        """
        Convert descriptor to a pandas Series.

        Returns
        -------
        pandas.Series
        """
        self.materialize()
        return pandas.Series(self._known_dtypes)

    def get_dtypes_set(self) -> set[np.dtype]:
        """
        Get a set of dtypes from the descriptor.

        Returns
        -------
        set[np.dtype]
        """
        if len(self._cols_with_unknown_dtypes) > 0 or not self._know_all_names:
            self._materialize_cols_with_unknown_dtypes()
        known_dtypes: set[np.dtype] = set(self._known_dtypes.values())
        if self._remaining_dtype is not None:
            known_dtypes.add(self._remaining_dtype)
        return known_dtypes

    @classmethod
    def _merge_dtypes(cls, values: list[Union['DtypesDescriptor', pandas.Series, None]]) -> 'DtypesDescriptor':
        """
        Union columns described by ``values`` and compute common dtypes for them.

        Parameters
        ----------
        values : list of DtypesDescriptors, pandas.Series or Nones

        Returns
        -------
        DtypesDescriptor
        """
        known_dtypes = {}
        cols_with_unknown_dtypes = []
        know_all_names = True
        dtypes_are_unknown = False
        dtypes_matrix = pandas.DataFrame()
        for i, val in enumerate(values):
            if isinstance(val, cls):
                know_all_names &= val._know_all_names
                dtypes = val._known_dtypes.copy()
                dtypes.update({col: 'unknown' for col in val._cols_with_unknown_dtypes})
                if val._remaining_dtype is not None:
                    know_all_names = False
                series = pandas.Series(dtypes, name=i)
                dtypes_matrix = pandas.concat([dtypes_matrix, series], axis=1)
                dtypes_matrix.fillna(value={i: pandas.api.types.pandas_dtype(float) if val._know_all_names and val._remaining_dtype is None else 'unknown'}, inplace=True)
            elif isinstance(val, pandas.Series):
                dtypes_matrix = pandas.concat([dtypes_matrix, val], axis=1)
            elif val is None:
                dtypes_are_unknown = True
                know_all_names = False
            else:
                raise NotImplementedError(type(val))
        if dtypes_are_unknown:
            return DtypesDescriptor(cols_with_unknown_dtypes=dtypes_matrix.index.tolist(), know_all_names=know_all_names)

        def combine_dtypes(row):
            if (row == 'unknown').any():
                return 'unknown'
            row = row.fillna(pandas.api.types.pandas_dtype('float'))
            return find_common_type(list(row.values))
        dtypes = dtypes_matrix.apply(combine_dtypes, axis=1)
        for col, dtype in dtypes.items():
            if dtype == 'unknown':
                cols_with_unknown_dtypes.append(col)
            else:
                known_dtypes[col] = dtype
        return DtypesDescriptor(known_dtypes, cols_with_unknown_dtypes, remaining_dtype=None, know_all_names=know_all_names)

    @classmethod
    def concat(cls, values: list[Union['DtypesDescriptor', pandas.Series, None]], axis: int=0) -> 'DtypesDescriptor':
        """
        Concatenate dtypes descriptors into a single descriptor.

        Parameters
        ----------
        values : list of DtypesDescriptors and pandas.Series
        axis : int, default: 0
            If ``axis == 0``: concatenate column names. This implements the logic of
            how dtypes are combined on ``pd.concat([df1, df2], axis=1)``.
            If ``axis == 1``: perform a union join for the column names described by
            `values` and then find common dtypes for the columns appeared to be in
            an intersection. This implements the logic of how dtypes are combined on
            ``pd.concat([df1, df2], axis=0).dtypes``.

        Returns
        -------
        DtypesDescriptor
        """
        if axis == 1:
            return cls._merge_dtypes(values)
        known_dtypes = {}
        cols_with_unknown_dtypes = []
        schema_is_known = True
        remaining_dtype = 'default'
        know_all_names = True
        for val in values:
            if isinstance(val, cls):
                all_new_cols = list(val._known_dtypes.keys()) + val._cols_with_unknown_dtypes
                if any((col in known_dtypes or col in cols_with_unknown_dtypes for col in all_new_cols)):
                    raise NotImplementedError('Duplicated column names are not yet supported by DtypesDescriptor')
                know_all_names &= val._know_all_names
                known_dtypes.update(val._known_dtypes)
                cols_with_unknown_dtypes.extend(val._cols_with_unknown_dtypes)
                if know_all_names:
                    if remaining_dtype == 'default' and val._remaining_dtype is not None:
                        remaining_dtype = val._remaining_dtype
                    elif remaining_dtype != 'default' and val._remaining_dtype is not None and (remaining_dtype != val._remaining_dtype):
                        remaining_dtype = None
                        know_all_names = False
                else:
                    remaining_dtype = None
                schema_is_known &= val._schema_is_known
            elif isinstance(val, pandas.Series):
                if any((col in known_dtypes or col in cols_with_unknown_dtypes for col in val.index)):
                    raise NotImplementedError('Duplicated column names are not yet supported by DtypesDescriptor')
                known_dtypes.update(val)
            elif val is None:
                remaining_dtype = None
                schema_is_known = False
                know_all_names = False
            else:
                raise NotImplementedError(type(val))
        return cls(known_dtypes, cols_with_unknown_dtypes, None if remaining_dtype == 'default' else remaining_dtype, parent_df=None, _schema_is_known=schema_is_known, know_all_names=know_all_names)

    @staticmethod
    def _normalize_levels(columns, reference=None):
        """
        Normalize levels of MultiIndex column names.

        The function fills missing levels with empty strings as pandas do:
        '''
        >>> columns = ["a", ("l1", "l2"), ("l1a", "l2a", "l3a")]
        >>> _normalize_levels(columns)
        [("a", "", ""), ("l1", "l2", ""), ("l1a", "l2a", "l3a")]
        >>> # with a reference
        >>> idx = pandas.MultiIndex(...)
        >>> idx.nlevels
        4
        >>> _normalize_levels(columns, reference=idx)
        [("a", "", "", ""), ("l1", "l2", "", ""), ("l1a", "l2a", "l3a", "")]
        '''

        Parameters
        ----------
        columns : sequence
            Labels to normalize. If dictionary, will replace keys with normalized columns.
        reference : pandas.Index, optional
            An index to match the number of levels with. If reference is a MultiIndex, then the reference number
            of levels should not be greater than the maximum number of levels in `columns`. If not specified,
            the `columns` themselves become a `reference`.

        Returns
        -------
        sequence
            Column values with normalized levels.
        dict[hashable, hashable]
            Mapping from old column names to new names, only contains column names that
            were changed.

        Raises
        ------
        ValueError
            When the reference number of levels is greater than the maximum number of levels
            in `columns`.
        """
        if reference is None:
            reference = columns
        if isinstance(reference, pandas.Index):
            max_nlevels = reference.nlevels
        else:
            max_nlevels = 1
            for col in reference:
                if isinstance(col, tuple):
                    max_nlevels = max(max_nlevels, len(col))
        if max_nlevels == 1:
            return (columns, {})
        max_columns_nlevels = 1
        for col in columns:
            if isinstance(col, tuple):
                max_columns_nlevels = max(max_columns_nlevels, len(col))
        if max_columns_nlevels > max_nlevels:
            raise ValueError(f'The reference number of levels is greater than the maximum number of levels in columns: {max_columns_nlevels} > {max_nlevels}')
        new_columns = []
        old_to_new_mapping = {}
        for col in columns:
            old_col = col
            if not isinstance(col, tuple):
                col = (col,)
            col = col + ('',) * (max_nlevels - len(col))
            new_columns.append(col)
            if old_col != col:
                old_to_new_mapping[old_col] = col
        return (new_columns, old_to_new_mapping)

    def _normalize_self_levels(self, reference=None):
        """
        Call ``self._normalize_levels()`` for known and unknown dtypes of this object.

        Parameters
        ----------
        reference : pandas.Index, optional
        """
        _, old_to_new_mapping = self._normalize_levels(self._known_dtypes.keys(), reference)
        for old_col, new_col in old_to_new_mapping.items():
            value = self._known_dtypes.pop(old_col)
            self._known_dtypes[new_col] = value
        self._cols_with_unknown_dtypes, _ = self._normalize_levels(self._cols_with_unknown_dtypes, reference)