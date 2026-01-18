import gymnasium as gym
import numpy as np
from typing import List, Optional, Tuple, Type, Union
import copy
import ray
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.algorithms.sac import SACTorchPolicy
from ray.rllib.algorithms.sac.rnnsac_torch_model import RNNSACTorchModel
from ray.rllib.algorithms.sac.sac_torch_policy import _get_dist_class
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import huber_loss, sequence_mask
from ray.rllib.utils.typing import ModelInputDict, TensorType, AlgorithmConfigDict
Constructs the loss for the Soft Actor Critic.

    Args:
        policy: The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[TorchDistributionWrapper]: The action distr. class.
        train_batch: The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    