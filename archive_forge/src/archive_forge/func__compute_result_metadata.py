import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.errors import MergeError
from modin.core.dataframe.base.dataframe.utils import join_columns
from modin.core.dataframe.pandas.metadata import ModinDtypes
from .utils import merge_partitioning
@classmethod
def _compute_result_metadata(cls, left, right, on, left_on, right_on, suffixes):
    """
        Compute columns and dtypes metadata for the result of merge if possible.

        Parameters
        ----------
        left : PandasQueryCompiler
        right : PandasQueryCompiler
        on : label, list of labels or None
            `on` argument that was passed to ``pandas.merge()``.
        left_on : label, list of labels or None
            `left_on` argument that was passed to ``pandas.merge()``.
        right_on : label, list of labels or None
            `right_on` argument that was passed to ``pandas.merge()``.
        suffixes : list of strings
            `suffixes` argument that was passed to ``pandas.merge()``.

        Returns
        -------
        new_columns : pandas.Index or None
            Columns for the result of merge. ``None`` if not enought metadata to compute.
        new_dtypes : ModinDtypes or None
            Dtypes for the result of merge. ``None`` if not enought metadata to compute.
        """
    new_columns = None
    new_dtypes = None
    if not left._modin_frame.has_materialized_columns:
        return (new_columns, new_dtypes)
    if left_on is None and right_on is None:
        if on is None:
            on = [c for c in left.columns if c in right.columns]
        _left_on, _right_on = (on, on)
    else:
        if left_on is None or right_on is None:
            raise MergeError("Must either pass only 'on' or 'left_on' and 'right_on', not combination of them.")
        _left_on, _right_on = (left_on, right_on)
    try:
        new_columns, left_renamer, right_renamer = join_columns(left.columns, right.columns, _left_on, _right_on, suffixes)
    except NotImplementedError:
        pass
    else:
        right_index_dtypes = right.index.dtypes if isinstance(right.index, pandas.MultiIndex) else pandas.Series([right.index.dtype], index=[right.index.name])
        right_dtypes = pandas.concat([right.dtypes, right_index_dtypes])[right_renamer.keys()].rename(right_renamer)
        left_index_dtypes = left._modin_frame._index_cache.maybe_get_dtypes()
        left_dtypes = ModinDtypes.concat([left._modin_frame._dtypes, left_index_dtypes]).lazy_get(left_renamer.keys()).set_index(list(left_renamer.values()))
        new_dtypes = ModinDtypes.concat([left_dtypes, right_dtypes])
    return (new_columns, new_dtypes)