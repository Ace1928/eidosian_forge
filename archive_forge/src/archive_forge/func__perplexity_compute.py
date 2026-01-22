from typing import Optional, Tuple
import torch
from torch import Tensor
def _perplexity_compute(total: Tensor, count: Tensor) -> Tensor:
    """Compute the Perplexity.

    Args:
        total: Log probabilities, summed over all samples
        count: Number of samples
    Returns:
        Perplexity

    """
    return torch.exp(total / count)