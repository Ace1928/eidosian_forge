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
def _load_baseline(lang: str='en', model_name_or_path: Optional[str]=None, baseline_path: Optional[str]=None, baseline_url: Optional[str]=None) -> Optional[Tensor]:
    """Load a CSV file with the baseline values used for rescaling."""
    if baseline_path:
        baseline: Optional[Tensor] = _read_csv_from_local_file(baseline_path)
    elif baseline_url:
        baseline = _read_csv_from_url(baseline_url)
    elif lang and model_name_or_path:
        url_base = 'https://raw.githubusercontent.com/Tiiiger/bert_score/master/bert_score/rescale_baseline'
        baseline_url = f'{url_base}/{lang}/{model_name_or_path}.tsv'
        baseline = _read_csv_from_url(baseline_url)
    else:
        rank_zero_warn('Baseline was not successfully loaded. No baseline is going to be used.')
        return None
    return baseline