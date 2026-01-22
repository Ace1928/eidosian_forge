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
def _get_mean_cell_probs(self, probabilities, segment_ids, row_ids, column_ids):
    """Computes average probability per cell, aggregating over tokens."""
    coords_to_probs = collections.defaultdict(list)
    for i, prob in self._get_cell_token_probs(probabilities, segment_ids, row_ids, column_ids):
        col = column_ids[i] - 1
        row = row_ids[i] - 1
        coords_to_probs[col, row].append(prob)
    return {coords: np.array(cell_probs).mean() for coords, cell_probs in coords_to_probs.items()}