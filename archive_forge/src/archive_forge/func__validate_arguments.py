import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
def _validate_arguments(self):
    bad_words_ids = self.bad_word_ids
    if not isinstance(bad_words_ids, list) or len(bad_words_ids) == 0:
        raise ValueError(f'`bad_words_ids` has to be a non-empty list, but is {bad_words_ids}.')
    if any((not isinstance(bad_word_ids, list) for bad_word_ids in bad_words_ids)):
        raise ValueError(f'`bad_words_ids` has to be a list of lists, but is {bad_words_ids}.')
    if any((any((not isinstance(token_id, (int, np.integer)) or token_id < 0 for token_id in bad_word_ids)) for bad_word_ids in bad_words_ids)):
        raise ValueError(f'Each list in `bad_words_ids` has to be a list of positive integers, but is {bad_words_ids}.')