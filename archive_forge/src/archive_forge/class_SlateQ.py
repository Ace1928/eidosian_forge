from typing import Any, Dict, List, Optional, Union
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
@Deprecated(old='rllib/algorithms/slate_q/', new='rllib_contrib/slate_q/', help=ALGO_DEPRECATION_WARNING, error=True)
class SlateQ(DQN):

    @classmethod
    @override(DQN)
    def get_default_config(cls) -> AlgorithmConfig:
        return SlateQConfig()