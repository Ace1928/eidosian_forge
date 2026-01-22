from typing import Optional
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING
@Deprecated(old='rllib/algorithms/ddppo/', new='rllib_contrib/ddppo/', help=ALGO_DEPRECATION_WARNING, error=True)
class DDPPO(PPO):

    @classmethod
    @override(PPO)
    def get_default_config(cls) -> AlgorithmConfig:
        return DDPPOConfig()