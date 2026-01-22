from itertools import permutations
from typing import Any, Callable, Tuple
import numpy as np
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE
def _find_best_perm_by_exhaustive_method(metric_mtx: Tensor, eval_func: Callable) -> Tuple[Tensor, Tensor]:
    """Solves the linear sum assignment problem using exhaustive method.

    This is done by exhaustively calculating the metric values of all possible permutations, and returns the best metric
    values and the corresponding permutations.

    Args:
        metric_mtx: the metric matrix, shape ``[batch_size, spk_num, spk_num]``
        eval_func: the function to reduce the metric values of different the permutations

    Returns:
        best_metric: shape ``[batch]``
        best_perm: shape ``[batch, spk]``

    """
    batch_size, spk_num = metric_mtx.shape[:2]
    ps = _gen_permutations(spk_num=spk_num, device=metric_mtx.device)
    perm_num = ps.shape[0]
    bps = ps.T[None, ...].expand(batch_size, spk_num, perm_num)
    metric_of_ps_details = torch.gather(metric_mtx, 2, bps)
    metric_of_ps = metric_of_ps_details.mean(dim=1)
    best_metric, best_indexes = eval_func(metric_of_ps, dim=1)
    best_indexes = best_indexes.detach()
    best_perm = ps[best_indexes, :]
    return (best_metric, best_perm)