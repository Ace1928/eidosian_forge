from typing import Any, Mapping
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.core.rl_module.marl_module import (
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.nested_dict import NestedDict
class BCTorchMultiAgentModuleWithSharedEncoder(MultiAgentRLModule):

    def __init__(self, config: MultiAgentRLModuleConfig) -> None:
        super().__init__(config)

    def setup(self):
        module_specs = self.config.modules
        module_spec = next(iter(module_specs.values()))
        global_dim = module_spec.observation_space['global'].shape[0]
        hidden_dim = module_spec.model_config_dict['fcnet_hiddens'][0]
        shared_encoder = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        rl_modules = {}
        for module_id, module_spec in module_specs.items():
            rl_modules[module_id] = module_spec.module_class(encoder=shared_encoder, local_dim=module_spec.observation_space['local'].shape[0], hidden_dim=hidden_dim, action_dim=module_spec.action_space.n)
        self._rl_modules = rl_modules

    def serialize(self):
        raise NotImplementedError

    def deserialize(self, data):
        raise NotImplementedError