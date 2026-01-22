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
def _infolm(preds: Union[str, Sequence[str]], target: Union[str, Sequence[str]], model_name_or_path: Union[str, os.PathLike]='bert-base-uncased', temperature: float=0.25, information_measure: _INFOLM_ALLOWED_INFORMATION_MEASURE_LITERAL='kl_divergence', idf: bool=True, alpha: Optional[float]=None, beta: Optional[float]=None, device: Optional[Union[str, torch.device]]=None, max_length: Optional[int]=None, batch_size: int=64, num_threads: int=0, verbose: bool=True, return_sentence_level_score: bool=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Wrapper for deprecated import.

    >>> preds = ['he read the book because he was interested in world history']
    >>> target = ['he was interested in world history because he read the book']
    >>> _infolm(preds, target, model_name_or_path='google/bert_uncased_L-2_H-128_A-2', idf=False)
    tensor(-0.1784)

    """
    _deprecated_root_import_func('infolm', 'text')
    return infolm(preds=preds, target=target, model_name_or_path=model_name_or_path, temperature=temperature, information_measure=information_measure, idf=idf, alpha=alpha, beta=beta, device=device, max_length=max_length, batch_size=batch_size, num_threads=num_threads, verbose=verbose, return_sentence_level_score=return_sentence_level_score)