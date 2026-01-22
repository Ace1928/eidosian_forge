import os
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics.functional.text.bert import bert_score
from torchmetrics.functional.text.bleu import bleu_score
from torchmetrics.functional.text.cer import char_error_rate
from torchmetrics.functional.text.chrf import chrf_score
from torchmetrics.functional.text.eed import extended_edit_distance
from torchmetrics.functional.text.infolm import (
from torchmetrics.functional.text.infolm import infolm
from torchmetrics.functional.text.mer import match_error_rate
from torchmetrics.functional.text.perplexity import perplexity
from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.functional.text.sacre_bleu import sacre_bleu_score
from torchmetrics.functional.text.squad import squad
from torchmetrics.functional.text.ter import translation_edit_rate
from torchmetrics.functional.text.wer import word_error_rate
from torchmetrics.functional.text.wil import word_information_lost
from torchmetrics.functional.text.wip import word_information_preserved
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_4
from torchmetrics.utilities.prints import _deprecated_root_import_func
def _extended_edit_distance(preds: Union[str, Sequence[str]], target: Sequence[Union[str, Sequence[str]]], language: Literal['en', 'ja']='en', return_sentence_level_score: bool=False, alpha: float=2.0, rho: float=0.3, deletion: float=0.2, insertion: float=1.0) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Wrapper for deprecated import.

    >>> preds = ["this is the prediction", "here is an other sample"]
    >>> target = ["this is the reference", "here is another one"]
    >>> _extended_edit_distance(preds=preds, target=target)
    tensor(0.3078)

    """
    _deprecated_root_import_func('extended_edit_distance', 'text')
    return extended_edit_distance(preds=preds, target=target, language=language, return_sentence_level_score=return_sentence_level_score, alpha=alpha, rho=rho, deletion=deletion, insertion=insertion)