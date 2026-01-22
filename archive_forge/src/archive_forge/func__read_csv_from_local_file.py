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
def _read_csv_from_local_file(baseline_path: str) -> Tensor:
    """Read baseline from csv file from the local file.

    This method implemented to avoid `pandas` dependency.

    """
    with open(baseline_path) as fname:
        csv_file = csv.reader(fname)
        baseline_list = [[float(item) for item in row] for idx, row in enumerate(csv_file) if idx > 0]
    return torch.tensor(baseline_list)[:, 1:]