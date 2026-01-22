from __future__ import annotations
from collections import (
from collections.abc import (
import csv
from io import StringIO
import re
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.inference import is_dict_like
from pandas.io.common import (
from pandas.io.parsers.base_parser import (
def _handle_usecols(self, columns: list[list[Scalar | None]], usecols_key: list[Scalar | None], num_original_columns: int) -> list[list[Scalar | None]]:
    """
        Sets self._col_indices

        usecols_key is used if there are string usecols.
        """
    col_indices: set[int] | list[int]
    if self.usecols is not None:
        if callable(self.usecols):
            col_indices = self._evaluate_usecols(self.usecols, usecols_key)
        elif any((isinstance(u, str) for u in self.usecols)):
            if len(columns) > 1:
                raise ValueError('If using multiple headers, usecols must be integers.')
            col_indices = []
            for col in self.usecols:
                if isinstance(col, str):
                    try:
                        col_indices.append(usecols_key.index(col))
                    except ValueError:
                        self._validate_usecols_names(self.usecols, usecols_key)
                else:
                    col_indices.append(col)
        else:
            missing_usecols = [col for col in self.usecols if col >= num_original_columns]
            if missing_usecols:
                raise ParserError(f'Defining usecols with out-of-bounds indices is not allowed. {missing_usecols} are out-of-bounds.')
            col_indices = self.usecols
        columns = [[n for i, n in enumerate(column) if i in col_indices] for column in columns]
        self._col_indices = sorted(col_indices)
    return columns