import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
def _shift_word_within_shifted_string(words: List[str], start: int, target: int, length: int) -> List[str]:
    shifted_words = words[:start]
    shifted_words += words[start + length:length + target]
    shifted_words += words[start:start + length]
    shifted_words += words[length + target:]
    return shifted_words