from gymnasium.spaces import Discrete, MultiDiscrete, Space
import numpy as np
from typing import Optional, Tuple, Union
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Categorical, MultiCategorical
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import NullContextManager
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.tf_utils import get_placeholder, one_hot as tf_one_hot
from ray.rllib.utils.torch_utils import one_hot
from ray.rllib.utils.typing import FromConfigSpec, ModelConfigDict, TensorType
def _postprocess_torch(self, policy, sample_batch):
    phis, _ = self.model._curiosity_feature_net({SampleBatch.OBS: torch.cat([torch.from_numpy(sample_batch[SampleBatch.OBS]).to(policy.device), torch.from_numpy(sample_batch[SampleBatch.NEXT_OBS]).to(policy.device)])})
    phi, next_phi = torch.chunk(phis, 2)
    actions_tensor = torch.from_numpy(sample_batch[SampleBatch.ACTIONS]).long().to(policy.device)
    predicted_next_phi = self.model._curiosity_forward_fcnet(torch.cat([phi, one_hot(actions_tensor, self.action_space).float()], dim=-1))
    forward_l2_norm_sqared = 0.5 * torch.sum(torch.pow(predicted_next_phi - next_phi, 2.0), dim=-1)
    forward_loss = torch.mean(forward_l2_norm_sqared)
    sample_batch[SampleBatch.REWARDS] = sample_batch[SampleBatch.REWARDS] + self.eta * forward_l2_norm_sqared.detach().cpu().numpy()
    phi_cat_next_phi = torch.cat([phi, next_phi], dim=-1)
    dist_inputs = self.model._curiosity_inverse_fcnet(phi_cat_next_phi)
    action_dist = TorchCategorical(dist_inputs, self.model) if isinstance(self.action_space, Discrete) else TorchMultiCategorical(dist_inputs, self.model, self.action_space.nvec)
    inverse_loss = -action_dist.logp(actions_tensor)
    inverse_loss = torch.mean(inverse_loss)
    loss = (1.0 - self.beta) * inverse_loss + self.beta * forward_loss
    self._optimizer.zero_grad()
    loss.backward()
    self._optimizer.step()
    return sample_batch