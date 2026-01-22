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
class SimpleArrowExtractor(BaseArrowExtractor[pa.Table, pa.Array, pa.Table]):

    def extract_row(self, pa_table: pa.Table) -> pa.Table:
        return pa_table

    def extract_column(self, pa_table: pa.Table) -> pa.Array:
        return pa_table.column(0)

    def extract_batch(self, pa_table: pa.Table) -> pa.Table:
        return pa_table