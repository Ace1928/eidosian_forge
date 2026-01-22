import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.imports import _NLTK_AVAILABLE
def _normalize_and_tokenize_text(text: str, stemmer: Optional[Any]=None, normalizer: Optional[Callable[[str], str]]=None, tokenizer: Optional[Callable[[str], Sequence[str]]]=None) -> Sequence[str]:
    """Rouge score should be calculated only over lowercased words and digits.

    Optionally, Porter stemmer can be used to strip word suffixes to improve matching. The text normalization follows
    the implemantion from `Rouge score_Text Normalizition`_.

    Args:
        text: An input sentence.
        stemmer: Porter stemmer instance to strip word suffixes to improve matching.
        normalizer: A user's own normalizer function.
            If this is ``None``, replacing any non-alpha-numeric characters with spaces is default.
            This function must take a ``str`` and return a ``str``.
        tokenizer:
            A user's own tokenizer function. If this is ``None``, splitting by spaces is default
            This function must take a ``str`` and return ``Sequence[str]``

    """
    text = normalizer(text) if callable(normalizer) else re.sub('[^a-z0-9]+', ' ', text.lower())
    tokens = tokenizer(text) if callable(tokenizer) else re.split('\\s+', text)
    if stemmer:
        tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens]
    return [x for x in tokens if isinstance(x, str) and len(x) > 0]