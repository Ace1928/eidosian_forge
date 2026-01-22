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
def _squad(preds: Union[Dict[str, str], List[Dict[str, str]]], target: SQUAD_TARGETS_TYPE) -> Dict[str, Tensor]:
    """Wrapper for deprecated import.

    >>> preds = [{"prediction_text": "1976", "id": "56e10a3be3433e1400422b22"}]
    >>> target = [{"answers": {"answer_start": [97], "text": ["1976"]},"id": "56e10a3be3433e1400422b22"}]
    >>> _squad(preds, target)
    {'exact_match': tensor(100.), 'f1': tensor(100.)}

    """
    _deprecated_root_import_func('squad', 'text')
    return squad(preds=preds, target=target)