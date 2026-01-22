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
def _parse_number(text):
    """Parses simple cardinal and ordinals numbers."""
    for suffix in _ORDINAL_SUFFIXES:
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    text = text.replace(',', '')
    try:
        value = float(text)
    except ValueError:
        return None
    if math.isnan(value):
        return None
    if value == _INF:
        return None
    return value