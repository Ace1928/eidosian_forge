from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics.functional.text.bert import bert_score
from torchmetrics.functional.text.helper_embedding_metric import _preprocess_text
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def _get_input_dict(input_ids: List[Tensor], attention_mask: List[Tensor]) -> Dict[str, Tensor]:
    """Create an input dictionary of ``input_ids`` and ``attention_mask`` for BERTScore calculation."""
    return {'input_ids': torch.cat(input_ids), 'attention_mask': torch.cat(attention_mask)}