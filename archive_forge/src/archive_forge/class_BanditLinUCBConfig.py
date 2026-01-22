from typing import Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING
class BanditLinUCBConfig(BanditConfig):

    def __init__(self):
        super().__init__(algo_class=BanditLinUCB)
        self.exploration_config = {'type': 'UpperConfidenceBound'}