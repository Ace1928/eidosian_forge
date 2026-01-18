import gymnasium as gym
from typing import Dict, List, Union
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
Initialize a TorchModelV2.

        Here is an example implementation for a subclass
        ``MyModelClass(TorchModelV2, nn.Module)``::

            def __init__(self, *args, **kwargs):
                TorchModelV2.__init__(self, *args, **kwargs)
                nn.Module.__init__(self)
                self._hidden_layers = nn.Sequential(...)
                self._logits = ...
                self._value_branch = ...
        