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
def _get_cell_token_probs(self, probabilities, segment_ids, row_ids, column_ids):
    for i, p in enumerate(probabilities):
        segment_id = segment_ids[i]
        col = column_ids[i] - 1
        row = row_ids[i] - 1
        if col >= 0 and row >= 0 and (segment_id == 1):
            yield (i, p)