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
def _chrf_score(preds: Union[str, Sequence[str]], target: Sequence[Union[str, Sequence[str]]], n_char_order: int=6, n_word_order: int=2, beta: float=2.0, lowercase: bool=False, whitespace: bool=False, return_sentence_level_score: bool=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Wrapper for deprecated import.

    >>> preds = ['the cat is on the mat']
    >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
    >>> _chrf_score(preds, target)
    tensor(0.8640)

    """
    _deprecated_root_import_func('chrf_score', 'text')
    return chrf_score(preds=preds, target=target, n_char_order=n_char_order, n_word_order=n_word_order, beta=beta, lowercase=lowercase, whitespace=whitespace, return_sentence_level_score=return_sentence_level_score)