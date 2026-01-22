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
def _get_table_boundaries(self, table):
    """Return maximal number of rows, columns and tokens."""
    max_num_tokens = 0
    max_num_columns = 0
    max_num_rows = 0
    for tc in table.selected_tokens:
        max_num_columns = max(max_num_columns, tc.column_index + 1)
        max_num_rows = max(max_num_rows, tc.row_index + 1)
        max_num_tokens = max(max_num_tokens, tc.token_index + 1)
        max_num_columns = min(self.max_column_id, max_num_columns)
        max_num_rows = min(self.max_row_id, max_num_rows)
    return (max_num_rows, max_num_columns, max_num_tokens)