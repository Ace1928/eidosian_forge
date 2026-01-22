from collections import Counter
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
def _bleu_score_compute(preds_len: Tensor, target_len: Tensor, numerator: Tensor, denominator: Tensor, n_gram: int, weights: Sequence[float], smooth: bool) -> Tensor:
    """Compute the BLEU score.

    Args:
        preds_len: count of words in a candidate translation
        target_len: count of words in a reference translation
        numerator: Numerator of precision score (true positives)
        denominator: Denominator of precision score (true positives + false positives)
        n_gram: gram value ranged 1 to 4
        weights: Weights used for unigrams, bigrams, etc. to calculate BLEU score.
        smooth: Whether to apply smoothing

    """
    device = numerator.device
    if min(numerator) == 0.0:
        return tensor(0.0, device=device)
    if smooth:
        precision_scores = torch.div(torch.add(numerator, torch.ones(n_gram, device=device)), torch.add(denominator, torch.ones(n_gram, device=device)))
        precision_scores[0] = numerator[0] / denominator[0]
    else:
        precision_scores = numerator / denominator
    log_precision_scores = tensor(weights, device=device) * torch.log(precision_scores)
    geometric_mean = torch.exp(torch.sum(log_precision_scores))
    brevity_penalty = tensor(1.0, device=device) if preds_len > target_len else torch.exp(1 - target_len / preds_len)
    return brevity_penalty * geometric_mean