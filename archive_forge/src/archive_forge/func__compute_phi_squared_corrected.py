from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.prints import rank_zero_warn
def _compute_phi_squared_corrected(phi_squared: Tensor, num_rows: int, num_cols: int, confmat_sum: Tensor) -> Tensor:
    """Compute bias-corrected Phi Squared."""
    return torch.max(torch.tensor(0.0, device=phi_squared.device), phi_squared - (num_rows - 1) * (num_cols - 1) / (confmat_sum - 1))