from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.prints import rank_zero_warn
def _compute_expected_freqs(confmat: Tensor) -> Tensor:
    """Compute the expected frequenceis from the provided confusion matrix."""
    margin_sum_rows, margin_sum_cols = (confmat.sum(1), confmat.sum(0))
    return torch.einsum('r, c -> rc', margin_sum_rows, margin_sum_cols) / confmat.sum()