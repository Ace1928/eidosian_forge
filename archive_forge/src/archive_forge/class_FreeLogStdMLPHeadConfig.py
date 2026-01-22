import abc
from dataclasses import dataclass, field
import functools
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
from ray.rllib.models.torch.misc import (
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import ExperimentalAPI
@ExperimentalAPI
@dataclass
class FreeLogStdMLPHeadConfig(_MLPConfig):
    """Configuration for an MLPHead with a floating second half of outputs.

    This model can be useful together with Gaussian Distributions.
    This gaussian distribution would be conditioned as follows:
        - The first half of outputs from this model can be used as
        state-dependent means when conditioning a gaussian distribution
        - The second half are floating free biases that can be used as
        state-independent standard deviations to condition a gaussian distribution.
    The mean values are produced by an MLPHead, while the standard
    deviations are added as floating free biases from a single 1D trainable variable
    (not dependent on the net's inputs).

    The output dimensions of the configured MLPHeadConfig must be even and are
    divided by two to gain the output dimensions of each the mean-net and the
    free std-variable.

    Example:
    .. testcode::
        :skipif: True

        # Configuration:
        config = FreeLogStdMLPHeadConfig(
            input_dims=[2],
            hidden_layer_dims=[16],
            hidden_layer_activation=None,
            hidden_layer_use_layernorm=False,
            hidden_layer_use_bias=True,
            output_layer_dim=8,  # <- this must be an even size
            output_layer_use_bias=True,
        )
        model = config.build(framework="tf2")

        # Resulting stack in pseudocode:
        # Linear(2, 16, bias=True)
        # Linear(8, 8, bias=True)  # 16 / 2 = 8 -> 8 nodes for the mean
        # Extra variable:
        # Tensor((8,), float32)  # for the free (observation independent) std outputs

    Example:
    .. testcode::
        :skipif: True

        # Configuration:
        config = FreeLogStdMLPHeadConfig(
            input_dims=[2],
            hidden_layer_dims=[31, 100],   # <- last idx must be an even size
            hidden_layer_activation="relu",
            hidden_layer_use_layernorm=False,
            hidden_layer_use_bias=False,
            output_layer_dim=None,  # use the last hidden layer as output layer
        )
        model = config.build(framework="torch")

        # Resulting stack in pseudocode:
        # Linear(2, 31, bias=False)
        # ReLu()
        # Linear(31, 50, bias=False)  # 100 / 2 = 50 -> 50 nodes for the mean
        # ReLu()
        # Extra variable:
        # Tensor((50,), float32)  # for the free (observation independent) std outputs
    """

    def _validate(self, framework: str='torch'):
        if len(self.output_dims) > 1 or self.output_dims[0] % 2 == 1:
            raise ValueError(f'`output_layer_dim` ({self.ouput_layer_dim}) or the last value in `hidden_layer_dims` ({self.hidden_layer_dims}) of a FreeLogStdMLPHeadConfig must be an even int (dividable by 2), e.g. `output_layer_dim=8` or `hidden_layer_dims=[133, 128]`!')

    @_framework_implemented()
    def build(self, framework: str='torch') -> 'Model':
        self._validate(framework=framework)
        if framework == 'torch':
            from ray.rllib.core.models.torch.heads import TorchFreeLogStdMLPHead
            return TorchFreeLogStdMLPHead(self)
        else:
            from ray.rllib.core.models.tf.heads import TfFreeLogStdMLPHead
            return TfFreeLogStdMLPHead(self)