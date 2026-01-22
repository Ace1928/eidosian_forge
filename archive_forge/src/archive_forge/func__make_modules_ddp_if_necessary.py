import logging
import pathlib
from typing import (
from ray.rllib.core.learner.learner import (
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModule
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchDDPRLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import (
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.torch_utils import (
from ray.rllib.utils.typing import Optimizer, Param, ParamDict, TensorType
@OverrideToImplementCustomLogic
def _make_modules_ddp_if_necessary(self) -> None:
    """Default logic for (maybe) making all Modules within self._module DDP."""
    if self._distributed:
        if isinstance(self._module, TorchRLModule):
            self._module = TorchDDPRLModule(self._module)
        else:
            assert isinstance(self._module, MultiAgentRLModule)
            for key in self._module.keys():
                sub_module = self._module[key]
                if isinstance(sub_module, TorchRLModule):
                    self._module.add_module(key, TorchDDPRLModule(sub_module), override=True)