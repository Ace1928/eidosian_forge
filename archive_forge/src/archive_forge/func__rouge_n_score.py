import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.imports import _NLTK_AVAILABLE
def _rouge_n_score(pred: Sequence[str], target: Sequence[str], n_gram: int) -> Dict[str, Tensor]:
    """Compute precision, recall and F1 score for the Rouge-N metric.

    Args:
        pred: A predicted sentence.
        target: A target sentence.
        n_gram: N-gram overlap.

    """

    def _create_ngrams(tokens: Sequence[str], n: int) -> Counter:
        ngrams: Counter = Counter()
        for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
            ngrams[ngram] += 1
        return ngrams
    pred_ngrams, target_ngrams = (_create_ngrams(pred, n_gram), _create_ngrams(target, n_gram))
    pred_len, target_len = (sum(pred_ngrams.values()), sum(target_ngrams.values()))
    if 0 in (pred_len, target_len):
        return {'precision': tensor(0.0), 'recall': tensor(0.0), 'fmeasure': tensor(0.0)}
    hits = sum((min(pred_ngrams[w], target_ngrams[w]) for w in set(pred_ngrams)))
    return _compute_metrics(hits, max(pred_len, 1), max(target_len, 1))