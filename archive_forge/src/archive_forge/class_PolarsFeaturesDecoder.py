import sys
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Optional
import pyarrow as pa
from .. import config
from ..features import Features
from ..features.features import decode_nested_example
from ..utils.py_utils import no_op_if_value_is_null
from .formatting import BaseArrowExtractor, TensorFormatter
class PolarsFeaturesDecoder:

    def __init__(self, features: Optional[Features]):
        self.features = features
        import polars as pl

    def decode_row(self, row: 'pl.DataFrame') -> 'pl.DataFrame':
        decode = {column_name: no_op_if_value_is_null(partial(decode_nested_example, feature)) for column_name, feature in self.features.items() if self.features._column_requires_decoding[column_name]} if self.features else {}
        if decode:
            row[list(decode.keys())] = row.map_rows(decode)
        return row

    def decode_column(self, column: 'pl.Series', column_name: str) -> 'pl.Series':
        decode = no_op_if_value_is_null(partial(decode_nested_example, self.features[column_name])) if self.features and column_name in self.features and self.features._column_requires_decoding[column_name] else None
        if decode:
            column = column.map_elements(decode)
        return column

    def decode_batch(self, batch: 'pl.DataFrame') -> 'pl.DataFrame':
        return self.decode_row(batch)