from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_resnet import ResNetConfig
class FlaxResNetBottleNeckLayer(nn.Module):
    """
    A classic ResNet's bottleneck layer composed by three `3x3` convolutions. The first `1x1` convolution reduces the
    input by a factor of `reduction` in order to make the second `3x3` convolution faster. The last `1x1` convolution
    remaps the reduced features to `out_channels`.
    """
    in_channels: int
    out_channels: int
    stride: int = 1
    activation: Optional[str] = 'relu'
    reduction: int = 4
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        should_apply_shortcut = self.in_channels != self.out_channels or self.stride != 1
        self.shortcut = FlaxResNetShortCut(self.out_channels, stride=self.stride, dtype=self.dtype) if should_apply_shortcut else None
        self.layer = FlaxResNetBottleNeckLayerCollection(self.out_channels, stride=self.stride, activation=self.activation, reduction=self.reduction, dtype=self.dtype)
        self.activation_func = ACT2FN[self.activation]

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        residual = hidden_state
        if self.shortcut is not None:
            residual = self.shortcut(residual, deterministic=deterministic)
        hidden_state = self.layer(hidden_state, deterministic)
        hidden_state += residual
        hidden_state = self.activation_func(hidden_state)
        return hidden_state