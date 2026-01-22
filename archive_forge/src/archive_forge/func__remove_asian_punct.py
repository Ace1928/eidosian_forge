import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
@classmethod
def _remove_asian_punct(cls: Type['_TercomTokenizer'], sentence: str) -> str:
    """Remove asian punctuation from an input sentence string."""
    sentence = re.sub(cls._ASIAN_PUNCTUATION, '', sentence)
    return re.sub(cls._FULL_WIDTH_PUNCTUATION, '', sentence)