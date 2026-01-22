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
class PythonArrowExtractor(BaseArrowExtractor[dict, list, dict]):

    def extract_row(self, pa_table: pa.Table) -> dict:
        return _unnest(pa_table.to_pydict())

    def extract_column(self, pa_table: pa.Table) -> list:
        return pa_table.column(0).to_pylist()

    def extract_batch(self, pa_table: pa.Table) -> dict:
        return pa_table.to_pydict()