import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
class CastError(ValueError):
    """When it's not possible to cast an Arrow table to a specific schema or set of features"""

    def __init__(self, *args, table_column_names: List[str], requested_column_names: List[str]) -> None:
        super().__init__(*args)
        self.table_column_names = table_column_names
        self.requested_column_names = requested_column_names

    def details(self):
        new_columns = set(self.table_column_names) - set(self.requested_column_names)
        missing_columns = set(self.requested_column_names) - set(self.table_column_names)
        if new_columns and missing_columns:
            return f'there are {len(new_columns)} new columns ({', '.join(new_columns)}) and {len(missing_columns)} missing columns ({', '.join(missing_columns)}).'
        elif new_columns:
            return f'there are {len(new_columns)} new columns ({new_columns})'
        else:
            return f'there are {len(missing_columns)} missing columns ({missing_columns})'