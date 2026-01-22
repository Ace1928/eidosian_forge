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
def apply_func(df):
    if has_external_grouper:
        external_grouper = df['grouper']
        external_grouper = [external_grouper.iloc[:, i] for i in range(len(external_grouper.columns))]
        for obj in external_grouper:
            if not isinstance(obj, pandas.Series):
                continue
            name = obj.name
            if isinstance(name, str):
                if name.startswith(MODIN_UNNAMED_SERIES_LABEL):
                    name = None
                elif name.endswith(duplicated_suffix):
                    name = re.sub(duplicated_pattern, '', name)
            elif isinstance(name, tuple):
                if name[-1].endswith(duplicated_suffix):
                    name = (*name[:-1], re.sub(duplicated_pattern, '', name[-1]))
            obj.name = name
        df = df['data']
    else:
        external_grouper = []
    by = []
    for idx in by_positions:
        if idx >= 0:
            by.append(external_grouper[idx])
        else:
            by.append(internal_by[-idx - 1])
    if series_groupby:
        df = df.squeeze(axis=1)
    if kwargs.get('level') is not None:
        assert len(by) == 0
        by = None
    result = operator(df.groupby(by, **kwargs))
    if align_result_columns and df.empty and result.empty:
        result.attrs[skip_on_aligning_flag] = True
    return result