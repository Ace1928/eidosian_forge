from typing import Dict, Any
from ray.rllib.models.utils import get_initializer
from ray.rllib.policy import Policy
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.annotations import is_overridden
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from gymnasium.spaces import Discrete
def _compute_action_probs(self, obs: TensorType) -> TensorType:
    """Compute action distribution over the action space.

        Args:
            obs: A tensor of observations of shape (batch_size * obs_dim)

        Returns:
            action_probs: A tensor of action probabilities
            of shape (batch_size * action_dim)
        """
    input_dict = {SampleBatch.OBS: obs}
    seq_lens = torch.ones(len(obs), device=self.device, dtype=int)
    state_batches = []
    if is_overridden(self.policy.action_distribution_fn):
        try:
            dist_inputs, dist_class, _ = self.policy.action_distribution_fn(self.policy.model, obs_batch=input_dict, state_batches=state_batches, seq_lens=seq_lens, explore=False, is_training=False)
        except TypeError:
            dist_inputs, dist_class, _ = self.policy.action_distribution_fn(self.policy, self.policy.model, input_dict=input_dict, state_batches=state_batches, seq_lens=seq_lens, explore=False, is_training=False)
    else:
        dist_class = self.policy.dist_class
        dist_inputs, _ = self.policy.model(input_dict, state_batches, seq_lens)
    action_dist = dist_class(dist_inputs, self.policy.model)
    assert isinstance(action_dist.dist, torch.distributions.categorical.Categorical), 'FQE only supports Categorical or MultiCategorical distributions!'
    action_probs = action_dist.dist.probs
    return action_probs