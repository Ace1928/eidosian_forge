from typing import List, Optional
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
@Deprecated(old='rllib/algorithms/maddpg/', new='rllib_contrib/maddpg/', help=ALGO_DEPRECATION_WARNING, error=True)
class MADDPG(DQN):

    @classmethod
    @override(DQN)
    def get_default_config(cls) -> AlgorithmConfig:
        return MADDPGConfig()