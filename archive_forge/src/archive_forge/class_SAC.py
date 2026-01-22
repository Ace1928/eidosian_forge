import logging
from typing import Type, Dict, Any, Optional, Union
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.algorithms.sac.sac_tf_policy import SACTFPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
class SAC(DQN):
    """Soft Actor Critic (SAC) Algorithm class.

    This file defines the distributed Algorithm class for the soft actor critic
    algorithm.
    See `sac_[tf|torch]_policy.py` for the definition of the policy loss.

    Detailed documentation:
    https://docs.ray.io/en/master/rllib-algorithms.html#sac
    """

    def __init__(self, *args, **kwargs):
        self._allow_unknown_subkeys += ['policy_model_config', 'q_model_config']
        super().__init__(*args, **kwargs)

    @classmethod
    @override(DQN)
    def get_default_config(cls) -> AlgorithmConfig:
        return SACConfig()

    @classmethod
    @override(DQN)
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Optional[Type[Policy]]:
        if config['framework'] == 'torch':
            from ray.rllib.algorithms.sac.sac_torch_policy import SACTorchPolicy
            return SACTorchPolicy
        else:
            return SACTFPolicy