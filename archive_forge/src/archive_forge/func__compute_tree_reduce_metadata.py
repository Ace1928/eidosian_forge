import datetime
import re
from typing import TYPE_CHECKING, Callable, Dict, Hashable, List, Optional, Union
import numpy as np
import pandas
from pandas._libs.lib import no_default
from pandas.api.types import is_object_dtype
from pandas.core.dtypes.common import is_dtype_equal, is_list_like, is_numeric_dtype
from pandas.core.indexes.api import Index, RangeIndex
from modin.config import Engine, IsRayCluster, MinPartitionSize, NPartitions
from modin.core.dataframe.base.dataframe.dataframe import ModinDataframe
from modin.core.dataframe.base.dataframe.utils import Axis, JoinType, is_trivial_index
from modin.core.dataframe.pandas.dataframe.utils import (
from modin.core.dataframe.pandas.metadata import (
from modin.core.storage_formats.pandas.parsers import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.storage_formats.pandas.utils import get_length_list
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.indexing import is_range_like
from modin.pandas.utils import check_both_not_none, is_full_grab_slice
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
def _compute_tree_reduce_metadata(self, axis, new_parts, dtypes=None):
    """
        Compute the metadata for the result of reduce function.

        Parameters
        ----------
        axis : int
            The axis on which reduce function was applied.
        new_parts : NumPy 2D array
            Partitions with the result of applied function.
        dtypes : str, optional
            The data types for the result. This is an optimization
            because there are functions that always result in a particular data
            type, and this allows us to avoid (re)computing it.

        Returns
        -------
        PandasDataframe
            Modin series (1xN frame) containing the reduced data.
        """
    new_axes, new_axes_lengths = ([0, 0], [0, 0])
    new_axes[axis] = [MODIN_UNNAMED_SERIES_LABEL]
    new_axes[axis ^ 1] = self.get_axis(axis ^ 1)
    new_axes_lengths[axis] = [1]
    new_axes_lengths[axis ^ 1] = self._get_axis_lengths(axis ^ 1)
    if dtypes == 'copy':
        dtypes = self.copy_dtypes_cache()
    elif dtypes is not None:
        dtypes = pandas.Series([pandas.api.types.pandas_dtype(dtypes)] * len(new_axes[1]), index=new_axes[1])
    result = self.__constructor__(new_parts, *new_axes, *new_axes_lengths, dtypes)
    return result