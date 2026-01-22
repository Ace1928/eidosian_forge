from collections.abc import Mapping, MutableMapping
from functools import partial
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, TypeVar, Union
import numpy as np
import pandas as pd
import pyarrow as pa
from packaging import version
from .. import config
from ..features import Features
from ..features.features import _ArrayXDExtensionType, _is_zero_copy_only, decode_nested_example, pandas_types_mapper
from ..table import Table
from ..utils.py_utils import no_op_if_value_is_null
class BaseArrowExtractor(Generic[RowFormat, ColumnFormat, BatchFormat]):
    """
    Arrow extractor are used to extract data from pyarrow tables.
    It makes it possible to extract rows, columns and batches.
    These three extractions types have to be implemented.
    """

    def extract_row(self, pa_table: pa.Table) -> RowFormat:
        raise NotImplementedError

    def extract_column(self, pa_table: pa.Table) -> ColumnFormat:
        raise NotImplementedError

    def extract_batch(self, pa_table: pa.Table) -> BatchFormat:
        raise NotImplementedError