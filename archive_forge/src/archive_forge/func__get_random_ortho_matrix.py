import math
from enum import Enum, auto
from typing import Optional
import torch
from torch.autograd.profiler import record_function
from .base import FeatureMap
@staticmethod
@torch.no_grad()
def _get_random_ortho_matrix(blocks: int, dim: int, device: torch.device, norm_distribution: NormDistribution=NormDistribution.Uniform) -> torch.Tensor:
    """
        Generate a random matrix whose rows are exactly orthonormal

        "How to generate random matrices from the classical compact groups", Mezzadri, 2007
        https://arxiv.org/pdf/math-ph/0609050v2.pdf

        .. note: the typical qr decomposition does not give uniform results, qr decomposition is not
        unique and the qr decomposition routines are biased towards numerical stability. See the above
        paper for more information.

        .. note: this does not follow the original implementation from the Performers authors.
        see docs/assets/kde plots to visualize the impact of using the R signs to correct Q
        """
    H = torch.randn((blocks, dim, dim), device=device, requires_grad=False)
    if norm_distribution == NormDistribution.Xi:
        norms = torch.sqrt(torch.einsum('...d,...d->...', H, H))
    Q, R = torch.linalg.qr(H)
    Q = torch.diag_embed(torch.sign(torch.diagonal(R, dim1=1, dim2=2))) @ Q
    if norm_distribution == NormDistribution.Xi:
        return torch.diag_embed(norms) @ Q
    return Q