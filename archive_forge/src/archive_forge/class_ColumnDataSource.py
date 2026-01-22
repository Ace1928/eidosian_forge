from __future__ import annotations
import logging # isort:skip
from typing import (
import numpy as np
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from ..util.deprecation import deprecated
from ..util.serialization import convert_datetime_array
from ..util.warnings import BokehUserWarning, warn
from .callbacks import CustomJS
from .filters import AllIndices, Filter, IntersectionFilter
from .selections import Selection, SelectionPolicy, UnionRenderers
class ColumnDataSource(ColumnarDataSource):
    """ Maps names of columns to sequences or arrays.

    The ``ColumnDataSource`` is a fundamental data structure of Bokeh. Most
    plots, data tables, etc. will be driven by a ``ColumnDataSource``.

    If the ``ColumnDataSource`` initializer is called with a single argument that
    can be any of the following:

    * A Python ``dict`` that maps string names to sequences of values, e.g.
      lists, arrays, etc.

      .. code-block:: python

          data = {'x': [1,2,3,4], 'y': np.array([10.0, 20.0, 30.0, 40.0])}

          source = ColumnDataSource(data)

    .. note::
        ``ColumnDataSource`` only creates a shallow copy of ``data``. Use e.g.
        ``ColumnDataSource(copy.deepcopy(data))`` if initializing from another
        ``ColumnDataSource.data`` object that you want to keep independent.

    * A Pandas ``DataFrame`` object

      .. code-block:: python

          source = ColumnDataSource(df)

      In this case the CDS will have columns corresponding to the columns of
      the ``DataFrame``. If the ``DataFrame`` columns have multiple levels,
      they will be flattened using an underscore (e.g. level_0_col_level_1_col).
      The index of the ``DataFrame`` will be flattened to an ``Index`` of tuples
      if it's a ``MultiIndex``, and then reset using ``reset_index``. The result
      will be a column with the same name if the index was named, or
      level_0_name_level_1_name if it was a named ``MultiIndex``. If the
      ``Index`` did not have a name or the ``MultiIndex`` name could not be
      flattened/determined, the ``reset_index`` function will name the index column
      ``index``, or ``level_0`` if the name ``index`` is not available.

    * A Pandas ``GroupBy`` object

      .. code-block:: python

          group = df.groupby(('colA', 'ColB'))

      In this case the CDS will have columns corresponding to the result of
      calling ``group.describe()``. The ``describe`` method generates columns
      for statistical measures such as ``mean`` and ``count`` for all the
      non-grouped original columns. The CDS columns are formed by joining
      original column names with the computed measure. For example, if a
      ``DataFrame`` has columns ``'year'`` and ``'mpg'``. Then passing
      ``df.groupby('year')`` to a CDS will result in columns such as
      ``'mpg_mean'``

      If the ``GroupBy.describe`` result has a named index column, then
      CDS will also have a column with this name. However, if the index name
      (or any subname of a ``MultiIndex``) is ``None``, then the CDS will have
      a column generically named ``index`` for the index.

      Note this capability to adapt ``GroupBy`` objects may only work with
      Pandas ``>=0.20.0``.

    .. note::
        There is an implicit assumption that all the columns in a given
        ``ColumnDataSource`` all have the same length at all times. For this
        reason, it is usually preferable to update the ``.data`` property
        of a data source "all at once".

    """
    data: DataDict = ColumnData(String, Seq(Any), help='\n    Mapping of column names to sequences of data. The columns can be, e.g,\n    Python lists or tuples, NumPy arrays, etc.\n\n    The .data attribute can also be set from Pandas DataFrames or GroupBy\n    objects. In these cases, the behaviour is identical to passing the objects\n    to the ``ColumnDataSource`` initializer.\n    ').accepts(Object('pandas.DataFrame'), lambda x: ColumnDataSource._data_from_df(x)).accepts(Object('pandas.core.groupby.GroupBy'), lambda x: ColumnDataSource._data_from_groupby(x)).asserts(lambda _, data: len({len(x) for x in data.values()}) <= 1, lambda obj, name, data: warn("ColumnDataSource's columns must be of the same length. " + 'Current lengths: %s' % ', '.join(sorted((str((k, len(v))) for k, v in data.items()))), BokehUserWarning))

    @overload
    def __init__(self, data: DataDict | pd.DataFrame | pd.core.groupby.GroupBy, **kwargs: TAny) -> None:
        ...

    @overload
    def __init__(self, **kwargs: TAny) -> None:
        ...

    def __init__(self, *args: TAny, **kwargs: TAny) -> None:
        """ If called with a single argument that is a dict or
        ``pandas.DataFrame``, treat that implicitly as the "data" attribute.

        """
        if len(args) == 1 and 'data' not in kwargs:
            kwargs['data'] = args[0]
        raw_data: DataDict = kwargs.pop('data', {})
        import pandas as pd
        if not isinstance(raw_data, dict):
            if isinstance(raw_data, pd.DataFrame):
                raw_data = self._data_from_df(raw_data)
            elif isinstance(raw_data, pd.core.groupby.GroupBy):
                raw_data = self._data_from_groupby(raw_data)
            else:
                raise ValueError(f'expected a dict or pandas.DataFrame, got {raw_data}')
        super().__init__(**kwargs)
        self.data.update(raw_data)

    @property
    def column_names(self) -> list[str]:
        """ A list of the column names in this data source.

        """
        return list(self.data)

    @staticmethod
    def _data_from_df(df: pd.DataFrame) -> DataDict:
        """ Create a ``dict`` of columns from a Pandas ``DataFrame``,
        suitable for creating a ColumnDataSource.

        Args:
            df (DataFrame) : data to convert

        Returns:
            dict[str, np.array]

        """
        import pandas as pd
        _df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            try:
                _df.columns = ['_'.join(col) for col in _df.columns.values]
            except TypeError:
                raise TypeError('Could not flatten MultiIndex columns. use string column names or flatten manually')
        if isinstance(df.columns, pd.CategoricalIndex):
            _df.columns = df.columns.tolist()
        index_name = ColumnDataSource._df_index_name(df)
        if index_name == 'index':
            _df.index = pd.Index(_df.index.values)
        else:
            _df.index = pd.Index(_df.index.values, name=index_name)
        _df.reset_index(inplace=True)
        tmp_data = {c: v.values for c, v in _df.items()}
        new_data: DataDict = {}
        for k, v in tmp_data.items():
            new_data[k] = v
        return new_data

    @staticmethod
    def _data_from_groupby(group: pd.core.groupby.GroupBy) -> DataDict:
        """ Create a ``dict`` of columns from a Pandas ``GroupBy``,
        suitable for creating a ``ColumnDataSource``.

        The data generated is the result of running ``describe``
        on the group.

        Args:
            group (GroupBy) : data to convert

        Returns:
            dict[str, np.array]

        """
        return ColumnDataSource._data_from_df(group.describe())

    @staticmethod
    def _df_index_name(df: pd.DataFrame) -> str:
        """ Return the Bokeh-appropriate column name for a ``DataFrame`` index

        If there is no named index, then `"index" is returned.

        If there is a single named index, then ``df.index.name`` is returned.

        If there is a multi-index, and the index names are all strings, then
        the names are joined with '_' and the result is returned, e.g. for a
        multi-index ``['ind1', 'ind2']`` the result will be "ind1_ind2".
        Otherwise if any index name is not a string, the fallback name "index"
        is returned.

        Args:
            df (DataFrame) : the ``DataFrame`` to find an index name for

        Returns:
            str

        """
        if df.index.name:
            return df.index.name
        elif df.index.names:
            try:
                return '_'.join(df.index.names)
            except TypeError:
                return 'index'
        else:
            return 'index'

    @classmethod
    def from_df(cls, data: pd.DataFrame) -> DataDict:
        """ Create a ``dict`` of columns from a Pandas ``DataFrame``,
        suitable for creating a ``ColumnDataSource``.

        Args:
            data (DataFrame) : data to convert

        Returns:
            dict[str, np.array]

        """
        return cls._data_from_df(data)

    @classmethod
    def from_groupby(cls, data: pd.core.groupby.GroupBy) -> DataDict:
        """ Create a ``dict`` of columns from a Pandas ``GroupBy``,
        suitable for creating a ``ColumnDataSource``.

        The data generated is the result of running ``describe``
        on the group.

        Args:
            data (Groupby) : data to convert

        Returns:
            dict[str, np.array]

        """
        return cls._data_from_df(data.describe())

    def to_df(self) -> pd.DataFrame:
        """ Convert this data source to pandas ``DataFrame``.

        Returns:
            DataFrame

        """
        import pandas as pd
        return pd.DataFrame(self.data)

    def add(self, data: Sequence[Any], name: str | None=None) -> str:
        """ Appends a new column of data to the data source.

        Args:
            data (seq) : new data to add
            name (str, optional) : column name to use.
                If not supplied, generate a name of the form "Series ####"

        Returns:
            str:  the column name used

        """
        if name is None:
            n = len(self.data)
            while f'Series {n}' in self.data:
                n += 1
            name = f'Series {n}'
        self.data[name] = data
        return name

    def remove(self, name: str) -> None:
        """ Remove a column of data.

        Args:
            name (str) : name of the column to remove

        Returns:
            None

        .. note::
            If the column name does not exist, a warning is issued.

        """
        try:
            del self.data[name]
        except (ValueError, KeyError):
            warn(f"Unable to find column '{name}' in data source")

    def stream(self, new_data: DataDict, rollover: int | None=None) -> None:
        """ Efficiently update data source columns with new append-only data.

        In cases where it is necessary to update data columns in, this method
        can efficiently send only the new data, instead of requiring the
        entire data set to be re-sent.

        Args:
            new_data (dict[str, seq]) : a mapping of column names to sequences of
                new data to append to each column.

                All columns of the data source must be present in ``new_data``,
                with identical-length append data.

            rollover (int, optional) : A maximum column size, above which data
                from the start of the column begins to be discarded. If None,
                then columns will continue to grow unbounded (default: None)

        Returns:
            None

        Raises:
            ValueError

        Example:

        .. code-block:: python

            source = ColumnDataSource(data=dict(foo=[], bar=[]))

            # has new, identical-length updates for all columns in source
            new_data = {
                'foo' : [10, 20],
                'bar' : [100, 200],
            }

            source.stream(new_data)

        """
        self._stream(new_data, rollover)

    def _stream(self, new_data: DataDict | pd.Series | pd.DataFrame, rollover: int | None=None, setter: Setter | None=None) -> None:
        """ Internal implementation to efficiently update data source columns
        with new append-only data. The internal implementation adds the setter
        attribute.  [https://github.com/bokeh/bokeh/issues/6577]

        In cases where it is necessary to update data columns in, this method
        can efficiently send only the new data, instead of requiring the
        entire data set to be re-sent.

        Args:
            new_data (dict[str, seq] or DataFrame or Series) : a mapping of
                column names to sequences of new data to append to each column,
                a pandas DataFrame, or a pandas Series in case of a single row -
                in this case the Series index is used as column names

                All columns of the data source must be present in ``new_data``,
                with identical-length append data.

            rollover (int, optional) : A maximum column size, above which data
                from the start of the column begins to be discarded. If None,
                then columns will continue to grow unbounded (default: None)
            setter (ClientSession or ServerSession or None, optional) :
                This is used to prevent "boomerang" updates to Bokeh apps.
                (default: None)
                In the context of a Bokeh server application, incoming updates
                to properties will be annotated with the session that is
                doing the updating. This value is propagated through any
                subsequent change notifications that the update triggers.
                The session can compare the event setter to itself, and
                suppress any updates that originate from itself.
        Returns:
            None

        Raises:
            ValueError

        Example:

        .. code-block:: python

            source = ColumnDataSource(data=dict(foo=[], bar=[]))

            # has new, identical-length updates for all columns in source
            new_data = {
                'foo' : [10, 20],
                'bar' : [100, 200],
            }

            source.stream(new_data)

        """
        import pandas as pd
        needs_length_check = True
        if isinstance(new_data, (pd.Series, pd.DataFrame)):
            if isinstance(new_data, pd.Series):
                new_data = new_data.to_frame().T
            needs_length_check = False
            _df = new_data
            newkeys = set(_df.columns)
            index_name = ColumnDataSource._df_index_name(_df)
            newkeys.add(index_name)
            new_data = dict(_df.items())
            new_data[index_name] = _df.index.values
        else:
            newkeys = set(new_data.keys())
        oldkeys = set(self.data.keys())
        if newkeys != oldkeys:
            missing = sorted(oldkeys - newkeys)
            extra = sorted(newkeys - oldkeys)
            if missing and extra:
                raise ValueError(f'Must stream updates to all existing columns (missing: {', '.join(missing)}, extra: {', '.join(extra)})')
            elif missing:
                raise ValueError(f'Must stream updates to all existing columns (missing: {', '.join(missing)})')
            else:
                raise ValueError(f'Must stream updates to all existing columns (extra: {', '.join(extra)})')
        if needs_length_check:
            lengths: set[int] = set()
            arr_types = (np.ndarray, pd.Series)
            for _, x in new_data.items():
                if isinstance(x, arr_types):
                    if len(x.shape) != 1:
                        raise ValueError(f'stream(...) only supports 1d sequences, got ndarray with size {x.shape!r}')
                    lengths.add(x.shape[0])
                else:
                    lengths.add(len(x))
            if len(lengths) > 1:
                raise ValueError('All streaming column updates must be the same length')
        for key, values in new_data.items():
            if pd and isinstance(values, (pd.Series, pd.Index)):
                values = values.values
            old_values = self.data[key]
            if isinstance(values, np.ndarray) and values.dtype.kind.lower() == 'm' and isinstance(old_values, np.ndarray) and (old_values.dtype.kind.lower() != 'm'):
                new_data[key] = convert_datetime_array(values)
            else:
                new_data[key] = values
        self.data._stream(self.document, self, new_data, rollover, setter)

    def patch(self, patches: Patches, setter: Setter | None=None) -> None:
        """ Efficiently update data source columns at specific locations

        If it is only necessary to update a small subset of data in a
        ``ColumnDataSource``, this method can be used to efficiently update only
        the subset, instead of requiring the entire data set to be sent.

        This method should be passed a dictionary that maps column names to
        lists of tuples that describe a patch change to apply. To replace
        individual items in columns entirely, the tuples should be of the
        form:

        .. code-block:: python

            (index, new_value)  # replace a single column value

            # or

            (slice, new_values) # replace several column values

        Values at an index or slice will be replaced with the corresponding
        new values.

        In the case of columns whose values are other arrays or lists, (e.g.
        image or patches glyphs), it is also possible to patch "subregions".
        In this case the first item of the tuple should be a whose first
        element is the index of the array item in the CDS patch, and whose
        subsequent elements are integer indices or slices into the array item:

        .. code-block:: python

            # replace the entire 10th column of the 2nd array:

              +----------------- index of item in column data source
              |
              |       +--------- row subindex into array item
              |       |
              |       |       +- column subindex into array item
              V       V       V
            ([2, slice(None), 10], new_values)

        Imagining a list of 2d NumPy arrays, the patch above is roughly
        equivalent to:

        .. code-block:: python

            data = [arr1, arr2, ...]  # list of 2d arrays

            data[2][:, 10] = new_data

        There are some limitations to the kinds of slices and data that can
        be accepted.

        * Negative ``start``, ``stop``, or ``step`` values for slices will
          result in a ``ValueError``.

        * In a slice, ``start > stop`` will result in a ``ValueError``

        * When patching 1d or 2d subitems, the subitems must be NumPy arrays.

        * New values must be supplied as a **flattened one-dimensional array**
          of the appropriate size.

        Args:
            patches (dict[str, list[tuple]]) : lists of patches for each column

        Returns:
            None

        Raises:
            ValueError

        Example:

        The following example shows how to patch entire column elements. In this case,

        .. code-block:: python

            source = ColumnDataSource(data=dict(foo=[10, 20, 30], bar=[100, 200, 300]))

            patches = {
                'foo' : [ (slice(2), [11, 12]) ],
                'bar' : [ (0, 101), (2, 301) ],
            }

            source.patch(patches)

        After this operation, the value of the ``source.data`` will be:

        .. code-block:: python

            dict(foo=[11, 12, 30], bar=[101, 200, 301])

        For a more comprehensive example, see :bokeh-tree:`examples/server/app/patch_app.py`.

        """
        extra = set(patches.keys()) - set(self.data.keys())
        if extra:
            raise ValueError('Can only patch existing columns (extra: %s)' % ', '.join(sorted(extra)))
        for name, patch in patches.items():
            col_len = len(self.data[name])
            for ind, _ in patch:
                if isinstance(ind, int):
                    if ind > col_len or ind < 0:
                        raise ValueError('Out-of bounds index (%d) in patch for column: %s' % (ind, name))
                elif isinstance(ind, slice):
                    _check_slice(ind)
                    if ind.stop is not None and ind.stop > col_len:
                        raise ValueError('Out-of bounds slice index stop (%d) in patch for column: %s' % (ind.stop, name))
                elif isinstance(ind, (list, tuple)):
                    if len(ind) == 0:
                        raise ValueError('Empty (length zero) patch multi-index')
                    if len(ind) == 1:
                        raise ValueError('Patch multi-index must contain more than one subindex')
                    ind_0 = ind[0]
                    if not isinstance(ind_0, int):
                        raise ValueError('Initial patch sub-index may only be integer, got: %s' % ind_0)
                    if ind_0 > col_len or ind_0 < 0:
                        raise ValueError('Out-of bounds initial sub-index (%d) in patch for column: %s' % (ind, name))
                    if not isinstance(self.data[name][ind_0], np.ndarray):
                        raise ValueError('Can only sub-patch into columns with NumPy array items')
                    if len(self.data[name][ind_0].shape) != len(ind) - 1:
                        raise ValueError('Shape mismatch between patch slice and sliced data')
                    elif isinstance(ind_0, slice):
                        _check_slice(ind_0)
                        if ind_0.stop is not None and ind_0.stop > col_len:
                            raise ValueError('Out-of bounds initial slice sub-index stop (%d) in patch for column: %s' % (ind.stop, name))
                    for subind in ind[1:]:
                        if not isinstance(subind, (int, slice)):
                            raise ValueError('Invalid patch sub-index: %s' % subind)
                        if isinstance(subind, slice):
                            _check_slice(subind)
                else:
                    raise ValueError('Invalid patch index: %s' % ind)
        self.data._patch(self.document, self, patches, setter)