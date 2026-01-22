from typing import Optional
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.a3c.a3c import A3CConfig, A3C
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING
@Deprecated(old='rllib/algorithms/a2c/', new='rllib_contrib/a2c/', help=ALGO_DEPRECATION_WARNING, error=True)
class A2C(A3C):

    @classmethod
    @override(A3C)
    def get_default_config(cls) -> AlgorithmConfig:
        return A2CConfig()