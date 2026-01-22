import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.imports import _NLTK_AVAILABLE
def _split_sentence(x: str) -> Sequence[str]:
    """Split sentence to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    if not _NLTK_AVAILABLE:
        raise ModuleNotFoundError('ROUGE-Lsum calculation requires that `nltk` is installed. Use `pip install nltk`.')
    import nltk
    _ensure_nltk_punkt_is_downloaded()
    re.sub('<n>', '', x)
    return nltk.sent_tokenize(x)