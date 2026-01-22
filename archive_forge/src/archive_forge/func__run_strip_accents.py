import collections
import os
import unicodedata
from typing import List, Optional, Tuple
from ....tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ....utils import logging
def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize('NFD', text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == 'Mn':
            continue
        output.append(char)
    return ''.join(output)