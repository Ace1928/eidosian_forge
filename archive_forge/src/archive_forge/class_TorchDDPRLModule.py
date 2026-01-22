import pathlib
from typing import Any, List, Mapping, Tuple, Union, Type
from packaging import version
from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.rl_module.rl_module_with_target_networks_interface import (
from ray.rllib.core.rl_module.torch.torch_compile_config import TorchCompileConfig
from ray.rllib.models.torch.torch_distributions import TorchDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import TORCH_COMPILE_REQUIRED_VERSION
from ray.rllib.utils.typing import NetworkType
class TorchDDPRLModule(RLModule, nn.parallel.DistributedDataParallel):

    def __init__(self, *args, **kwargs) -> None:
        nn.parallel.DistributedDataParallel.__init__(self, *args, **kwargs)
        self.config = self.unwrapped().config

    def get_train_action_dist_cls(self, *args, **kwargs) -> Type[TorchDistribution]:
        return self.unwrapped().get_train_action_dist_cls(*args, **kwargs)

    def get_exploration_action_dist_cls(self, *args, **kwargs) -> Type[TorchDistribution]:
        return self.unwrapped().get_exploration_action_dist_cls(*args, **kwargs)

    def get_inference_action_dist_cls(self, *args, **kwargs) -> Type[TorchDistribution]:
        return self.unwrapped().get_inference_action_dist_cls(*args, **kwargs)

    @override(RLModule)
    def _forward_train(self, *args, **kwargs):
        return self(*args, **kwargs)

    @override(RLModule)
    def _forward_inference(self, *args, **kwargs) -> Mapping[str, Any]:
        return self.unwrapped()._forward_inference(*args, **kwargs)

    @override(RLModule)
    def _forward_exploration(self, *args, **kwargs) -> Mapping[str, Any]:
        return self.unwrapped()._forward_exploration(*args, **kwargs)

    @override(RLModule)
    def get_state(self, *args, **kwargs):
        return self.unwrapped().get_state(*args, **kwargs)

    @override(RLModule)
    def set_state(self, *args, **kwargs):
        self.unwrapped().set_state(*args, **kwargs)

    @override(RLModule)
    def save_state(self, *args, **kwargs):
        self.unwrapped().save_state(*args, **kwargs)

    @override(RLModule)
    def load_state(self, *args, **kwargs):
        self.unwrapped().load_state(*args, **kwargs)

    @override(RLModule)
    def save_to_checkpoint(self, *args, **kwargs):
        self.unwrapped().save_to_checkpoint(*args, **kwargs)

    @override(RLModule)
    def _save_module_metadata(self, *args, **kwargs):
        self.unwrapped()._save_module_metadata(*args, **kwargs)

    @override(RLModule)
    def _module_metadata(self, *args, **kwargs):
        return self.unwrapped()._module_metadata(*args, **kwargs)

    @override(RLModule)
    def unwrapped(self) -> 'RLModule':
        return self.module