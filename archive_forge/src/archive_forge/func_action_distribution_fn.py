import gymnasium as gym
from typing import Dict, Union, List, Tuple, Optional
import numpy as np
from ray.rllib.policy.policy import Policy, ViewRequirement
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils.typing import AlgorithmConfigDict, TensorStructType, TensorType
from ray.rllib.utils.annotations import override
from ray.rllib.utils.debug import update_global_seed_if_necessary
def action_distribution_fn(self, model, obs_batch: TensorStructType, **kwargs) -> Tuple[TensorType, type, List[TensorType]]:
    obs = np.array(obs_batch[SampleBatch.OBS], dtype=int)
    action_probs = self.action_dist[obs]
    with np.errstate(divide='ignore'):
        return (np.log(action_probs), TorchCategorical, None)