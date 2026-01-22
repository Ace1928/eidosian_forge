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
class TorchDDPRLModuleWithTargetNetworksInterface(TorchDDPRLModule, RLModuleWithTargetNetworksInterface):

    @override(RLModuleWithTargetNetworksInterface)
    def get_target_network_pairs(self) -> List[Tuple[NetworkType, NetworkType]]:
        return self.module.get_target_network_pairs()