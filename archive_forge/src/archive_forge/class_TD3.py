from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ddpg.ddpg import DDPG, DDPGConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
@Deprecated(old='rllib/algorithms/td3/', new='rllib_contrib/td3/', help=ALGO_DEPRECATION_WARNING, error=True)
class TD3(DDPG):

    @classmethod
    @override(DDPG)
    def get_default_config(cls) -> AlgorithmConfig:
        return TD3Config()