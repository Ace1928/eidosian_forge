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
class PolarsArrowExtractor(BaseArrowExtractor['pl.DataFrame', 'pl.Series', 'pl.DataFrame']):

    def extract_row(self, pa_table: pa.Table) -> 'pl.DataFrame':
        if config.POLARS_AVAILABLE:
            if 'polars' not in sys.modules:
                import polars
            else:
                polars = sys.modules['polars']
            return polars.from_arrow(pa_table.slice(length=1))
        else:
            raise ValueError('Polars needs to be installed to be able to return Polars dataframes.')

    def extract_column(self, pa_table: pa.Table) -> 'pl.Series':
        if config.POLARS_AVAILABLE:
            if 'polars' not in sys.modules:
                import polars
            else:
                polars = sys.modules['polars']
            return polars.from_arrow(pa_table.select([0]))[pa_table.column_names[0]]
        else:
            raise ValueError('Polars needs to be installed to be able to return Polars dataframes.')

    def extract_batch(self, pa_table: pa.Table) -> 'pl.DataFrame':
        if config.POLARS_AVAILABLE:
            if 'polars' not in sys.modules:
                import polars
            else:
                polars = sys.modules['polars']
            return polars.from_arrow(pa_table)
        else:
            raise ValueError('Polars needs to be installed to be able to return Polars dataframes.')