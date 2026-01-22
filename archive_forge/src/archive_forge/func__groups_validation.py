from typing import Dict, List, Optional, Tuple
import torch
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import _flexible_bincount
def _groups_validation(groups: torch.Tensor, num_groups: int) -> None:
    """Validate groups tensor.

    - The largest number in the tensor should not be larger than the number of groups. The group identifiers should
    be ``0, 1, ..., (num_groups - 1)``.
    - The group tensor should be dtype long.

    """
    if torch.max(groups) > num_groups:
        raise ValueError(f'The largest number in the groups tensor is {torch.max(groups)}, which is larger than the specified', f'number of groups {num_groups}. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.')
    if groups.dtype != torch.long:
        raise ValueError(f'Expected dtype of argument groups to be long, not {groups.dtype}.')