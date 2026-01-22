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
def _get_column_values(self, table, col_index):
    table_numeric_values = {}
    for row_index, row in table.iterrows():
        cell = row[col_index]
        if cell.numeric_value is not None:
            table_numeric_values[row_index] = cell.numeric_value
    return table_numeric_values