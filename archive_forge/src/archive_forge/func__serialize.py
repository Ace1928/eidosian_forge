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
def _serialize(self, question_tokens, table, num_columns, num_rows, num_tokens):
    """Serializes table and text."""
    tokens, segment_ids, column_ids, row_ids = self._serialize_text(question_tokens)
    tokens.append(self.sep_token)
    segment_ids.append(0)
    column_ids.append(0)
    row_ids.append(0)
    for token, column_id, row_id in self._get_table_values(table, num_columns, num_rows, num_tokens):
        tokens.append(token)
        segment_ids.append(1)
        column_ids.append(column_id)
        row_ids.append(row_id)
    return SerializedExample(tokens=tokens, segment_ids=segment_ids, column_ids=column_ids, row_ids=row_ids)