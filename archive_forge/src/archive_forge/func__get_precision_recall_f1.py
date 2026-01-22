import csv
import urllib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchmetrics.functional.text.helper_embedding_metric import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
def _get_precision_recall_f1(preds_embeddings: Tensor, target_embeddings: Tensor, preds_idf_scale: Tensor, target_idf_scale: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Calculate precision, recall and F1 score over candidate and reference sentences.

    Args:
        preds_embeddings: Embeddings of candidate sentences.
        target_embeddings: Embeddings of reference sentences.
        preds_idf_scale: An IDF scale factor for candidate sentences.
        target_idf_scale: An IDF scale factor for reference sentences.

    Return:
        Tensors containing precision, recall and F1 score, respectively.

    """
    cos_sim = torch.einsum('blpd, blrd -> blpr', preds_embeddings, target_embeddings)
    precision = _get_scaled_precision_or_recall(cos_sim, 'precision', preds_idf_scale)
    recall = _get_scaled_precision_or_recall(cos_sim, 'recall', target_idf_scale)
    f1_score = 2 * precision * recall / (precision + recall)
    f1_score = f1_score.masked_fill(torch.isnan(f1_score), 0.0)
    return (precision, recall, f1_score)