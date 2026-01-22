import math
from enum import Enum, auto
from typing import Optional
import torch
from torch.autograd.profiler import record_function
from .base import FeatureMap
class SMHyperbolic(SoftMaxPositiveEstimators):
    """
    "Positive random features hyperbolic" estimator, SMHyp+,
    as proposed in the Performers_ paper, Lemma 1.

    _Performers: "Rethinking attention with performers." K. Choromanski et al. (2020).
    https://arxiv.org/pdf/2009.14794v1.pdf
    """

    def __init__(self, dim_features: int, iter_before_redraw: Optional[int], normalize_inputs: bool=False, epsilon: float=1e-06, softmax_temp: float=-1):
        super().__init__(dim_features, iter_before_redraw, normalize_inputs, epsilon, softmax_temp)
        assert dim_features % 2 == 0, 'The feature dimension needs to be even with this kernel'
        self.dim_feature_map = self.dim_features // 2

    @torch.no_grad()
    def _get_feature_map(self, dim_input: int, dim_features: int, device: torch.device):
        """
        Generate the projection matrix onto the random features

        .. note: The heads dimension needs to be taken into account, hence the per-block random matrix
        and not uniformally random.
        """
        features = self._get_random_ortho_matrix(math.ceil(dim_input / dim_features), dim_features, norm_distribution=NormDistribution.Xi, device=device)
        return features.flatten(0, 1)[:dim_input]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = super().pre_scale(x)
        x_scaled = x_scaled @ self.features
        return torch.cat([torch.exp(x_scaled + self.offset), torch.exp(-x_scaled + self.offset)], dim=-1)