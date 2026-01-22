import ast
import hashlib
import re
import warnings
from collections.abc import Iterable
from typing import Hashable, List
import numpy as np
import pandas
from pandas._libs import lib
from pandas.api.types import is_scalar
from pandas.core.apply import reconstruct_func
from pandas.core.common import is_bool_indexer
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.groupby.base import transformation_kernels
from pandas.core.indexes.api import ensure_index_from_sequences
from pandas.core.indexing import check_bool_indexer
from pandas.errors import DataError
from modin.config import CpuCount, RangePartitioning, use_range_partitioning_groupby
from modin.core.dataframe.algebra import (
from modin.core.dataframe.algebra.default2pandas.groupby import (
from modin.core.dataframe.pandas.metadata import (
from modin.core.storage_formats import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import get_logger
from modin.utils import (
from .aggregations import CorrCovBuilder
from .groupby import GroupbyReduceImpl, PivotTableImpl
from .merge import MergeImpl
from .utils import get_group_names, merge_partitioning
def _dt_prop_map(property_name):
    """
    Build function that access specified property of the ``dt`` property of the passed frame.

    Parameters
    ----------
    property_name : str
        Date-time property name to access.

    Returns
    -------
    callable(pandas.DataFrame, *args, **kwargs) -> pandas.DataFrame
        Function to be applied in the partitions.

    Notes
    -----
    This applies non-callable properties of ``Series.dt``.
    """

    def dt_op_builder(df, *args, **kwargs):
        """Access specified date-time property of the passed frame."""
        squeezed_df = df.squeeze(axis=1)
        if isinstance(squeezed_df, pandas.DataFrame) and len(squeezed_df.columns) == 0:
            return squeezed_df
        assert isinstance(squeezed_df, pandas.Series)
        prop_val = getattr(squeezed_df.dt, property_name)
        if isinstance(prop_val, pandas.Series):
            return prop_val.to_frame()
        elif isinstance(prop_val, pandas.DataFrame):
            return prop_val
        else:
            return pandas.DataFrame([prop_val])
    return dt_op_builder