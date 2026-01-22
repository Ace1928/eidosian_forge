import collections
from typing import Any, Dict, Iterable, Optional, Sequence
import numpy as np
import pyarrow as pa
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.error_message import ErrorMessage
from modin.experimental.core.execution.native.implementations.hdk_on_native.dataframe.dataframe import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.df_algebra import (
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .column import HdkProtocolColumn
from .utils import raise_copy_alert_if_materialize
def _yield_chunks(self, chunk_slices) -> 'HdkProtocolDataframe':
    """
        Yield DataFrame chunks according to the passed offsets.

        Parameters
        ----------
        chunk_slices : list
            Chunking offsets.

        Yields
        ------
        HdkProtocolDataframe
        """
    for i in range(len(chunk_slices) - 1):
        yield HdkProtocolDataframe(df=self._df.take_2d_labels_or_positional(row_positions=range(chunk_slices[i], chunk_slices[i + 1])), nan_as_null=self._nan_as_null, allow_copy=self._allow_copy)