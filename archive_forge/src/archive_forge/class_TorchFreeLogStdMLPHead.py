import functools
from typing import Optional
import numpy as np
from ray.rllib.core.models.base import Model
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.specs.checker import SpecCheckingError
from ray.rllib.core.models.specs.specs_base import Spec
from ray.rllib.core.models.specs.specs_base import TensorSpec
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.torch.primitives import TorchCNNTranspose, TorchMLP
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
class TorchFreeLogStdMLPHead(TorchModel):
    """An MLPHead that implements floating log stds for Gaussian distributions."""

    def __init__(self, config: FreeLogStdMLPHeadConfig) -> None:
        super().__init__(config)
        assert config.output_dims[0] % 2 == 0, 'output_dims must be even for free std!'
        self._half_output_dim = config.output_dims[0] // 2
        self.net = TorchMLP(input_dim=config.input_dims[0], hidden_layer_dims=config.hidden_layer_dims, hidden_layer_activation=config.hidden_layer_activation, hidden_layer_use_layernorm=config.hidden_layer_use_layernorm, hidden_layer_use_bias=config.hidden_layer_use_bias, output_dim=self._half_output_dim, output_activation=config.output_layer_activation, output_use_bias=config.output_layer_use_bias)
        self.log_std = torch.nn.Parameter(torch.as_tensor([0.0] * self._half_output_dim))

    @override(Model)
    def get_input_specs(self) -> Optional[Spec]:
        return TensorSpec('b, d', d=self.config.input_dims[0], framework='torch')

    @override(Model)
    def get_output_specs(self) -> Optional[Spec]:
        return TensorSpec('b, d', d=self.config.output_dims[0], framework='torch')

    @override(Model)
    @auto_fold_unfold_time('input_specs')
    def _forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        mean = self.net(inputs)
        return torch.cat([mean, self.log_std.unsqueeze(0).repeat([len(mean), 1])], axis=1)