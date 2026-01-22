from math import ceil
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Tuple
import numpy as np
import pandas
import pyarrow as pa
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.core.dataframe.base.interchange.dataframe_protocol.utils import (
from modin.utils import _inherit_docstrings
from .buffer import HdkProtocolBuffer
from .utils import arrow_dtype_to_arrow_c, arrow_types_map
def _materialize_actual_buffers(self):
    """
        Materialize PyArrow table's buffers that can be zero-copy returned to a consumer, if they aren't already materialized.

        Besides materializing PyArrow table itself (if there were some delayed computations)
        the function also may do the following if required:
        1. Propagate external dtypes to the PyArrow table. For example,
            if ``self.dtype`` is a string kind, but internal PyArrow dtype is a dictionary
            (if the table were just exported from HDK), then the dictionary will be casted
            to string dtype.
        2. Combine physical chunks of PyArrow table into a single contiguous buffer.
        """
    if self.num_chunks() != 1:
        if not self._col._allow_copy:
            raise_copy_alert(copy_reason='physical chunks combining due to contiguous buffer materialization')
        self._combine_chunks()
    external_dtype = self.dtype
    internal_dtype = self._dtype_from_pyarrow(self._arrow_dtype)
    if external_dtype[0] != internal_dtype[0]:
        self._propagate_dtype(external_dtype)