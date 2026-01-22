import re
from typing import Hashable, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow
from pandas._libs.lib import no_default
from pandas.core.dtypes.common import (
from pandas.core.indexes.api import Index, MultiIndex, RangeIndex
from pyarrow.types import is_dictionary
from modin.core.dataframe.base.dataframe.utils import (
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from modin.core.dataframe.pandas.metadata import LazyProxyCategoricalDtype
from modin.core.dataframe.pandas.metadata.dtypes import get_categories_dtype
from modin.core.dataframe.pandas.utils import concatenate
from modin.error_message import ErrorMessage
from modin.experimental.core.storage_formats.hdk.query_compiler import (
from modin.pandas.indexing import is_range_like
from modin.pandas.utils import check_both_not_none
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, _inherit_docstrings
from ..db_worker import DbTable
from ..df_algebra import (
from ..expr import (
from ..partitioning.partition_manager import HdkOnNativeDataframePartitionManager
from .utils import (
def _materialize_rowid(self):
    """
        Materialize virtual 'rowid' column.

        Make a projection with a virtual 'rowid' column materialized
        as '__index__' column.

        Returns
        -------
        HdkOnNativeDataframe
            The new frame.
        """
    name = self._index_cache.get().name if self.has_materialized_index else None
    name = mangle_index_names([name])[0]
    exprs = dict()
    exprs[name] = self.ref(ROWID_COL_NAME)
    for col in self._table_cols:
        exprs[col] = self.ref(col)
    return self.__constructor__(columns=self.columns, dtypes=self._dtypes_for_exprs(exprs), op=TransformNode(self, exprs), index_cols=[name], uses_rowid=True, force_execution_mode=self._force_execution_mode)