import json
import os
import sys
import warnings
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken, BatchEncoding
from ...utils import (
@staticmethod
def _get_word_offsets(offsets: Dict[str, Union[str, float]], word_delimiter_char: str=' ') -> Dict[str, Union[str, float]]:
    word_offsets = []
    last_state = 'SPACE'
    word = ''
    start_offset = 0
    end_offset = 0
    for i, offset in enumerate(offsets):
        char = offset['char']
        state = 'SPACE' if char == word_delimiter_char else 'WORD'
        if state == last_state:
            end_offset = offset['end_offset']
            word += char
        elif state == 'SPACE':
            word_offsets.append({'word': word, 'start_offset': start_offset, 'end_offset': end_offset})
        else:
            start_offset = offset['start_offset']
            end_offset = offset['end_offset']
            word = char
        last_state = state
    if last_state == 'WORD':
        word_offsets.append({'word': word, 'start_offset': start_offset, 'end_offset': end_offset})
    return word_offsets