import torch
from torch import Tensor
from typing_extensions import Literal
def _fleiss_kappa_compute(counts: Tensor) -> Tensor:
    """Computes fleiss kappa from counts matrix.

    Args:
        counts: counts matrix of shape [n_samples, n_categories]

    """
    total = counts.shape[0]
    num_raters = counts.sum(1).max()
    p_i = counts.sum(dim=0) / (total * num_raters)
    p_j = ((counts ** 2).sum(dim=1) - num_raters) / (num_raters * (num_raters - 1))
    p_bar = p_j.mean()
    pe_bar = (p_i ** 2).sum()
    return (p_bar - pe_bar) / (1 - pe_bar + 1e-05)