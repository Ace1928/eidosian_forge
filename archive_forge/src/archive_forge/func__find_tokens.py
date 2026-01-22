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
def _find_tokens(self, text, segment):
    """Return start index of segment in text or None."""
    logging.info(f'text: {text} {segment}')
    for index in range(1 + len(text) - len(segment)):
        for seg_index, seg_token in enumerate(segment):
            if text[index + seg_index].piece != seg_token.piece:
                break
        else:
            return index
    return None