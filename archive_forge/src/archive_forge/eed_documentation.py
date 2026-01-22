import re
import unicodedata
from math import inf
from typing import List, Optional, Sequence, Tuple, Union
from torch import Tensor, stack, tensor
from typing_extensions import Literal
from torchmetrics.functional.text.helper import _validate_inputs
Compute extended edit distance score (`ExtendedEditDistance`_) [1] for strings or list of strings.

    The metric utilises the Levenshtein distance and extends it by adding a jump operation.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en
        return_sentence_level_score: An indication of whether sentence-level EED score is to be returned.
        alpha: optimal jump penalty, penalty for jumps between characters
        rho: coverage cost, penalty for repetition of characters
        deletion: penalty for deletion of character
        insertion: penalty for insertion or substitution of character

    Return:
        Extended edit distance score as a tensor

    Example:
        >>> from torchmetrics.functional.text import extended_edit_distance
        >>> preds = ["this is the prediction", "here is an other sample"]
        >>> target = ["this is the reference", "here is another one"]
        >>> extended_edit_distance(preds=preds, target=target)
        tensor(0.3078)

    References:
        [1] P. Stanchev, W. Wang, and H. Ney, “EED: Extended Edit Distance Measure for Machine Translation”,
        submitted to WMT 2019. `ExtendedEditDistance`_

    