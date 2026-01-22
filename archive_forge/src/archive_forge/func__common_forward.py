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
def _common_forward(self, batch):
    obs = batch['obs']
    global_enc = self.encoder(obs['global'])
    policy_in = torch.cat([global_enc, obs['local']], dim=-1)
    action_logits = self.policy_head(policy_in)
    return {SampleBatch.ACTION_DIST_INPUTS: action_logits}