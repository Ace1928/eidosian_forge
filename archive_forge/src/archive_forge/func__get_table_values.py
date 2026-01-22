import collections
import datetime
import enum
import itertools
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Text, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import (
from ...utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available, logging
def _get_table_values(self, table, num_columns, num_rows, num_tokens) -> Generator[TableValue, None, None]:
    """Iterates over partial table and returns token, column and row indexes."""
    for tc in table.selected_tokens:
        if tc.row_index >= num_rows + 1:
            continue
        if tc.column_index >= num_columns:
            continue
        cell = table.rows[tc.row_index][tc.column_index]
        token = cell[tc.token_index]
        word_begin_index = tc.token_index
        while word_begin_index >= 0 and _is_inner_wordpiece(cell[word_begin_index]):
            word_begin_index -= 1
        if word_begin_index >= num_tokens:
            continue
        yield TableValue(token, tc.column_index + 1, tc.row_index)