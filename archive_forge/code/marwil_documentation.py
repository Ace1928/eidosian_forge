from typing import Callable, Optional, Type, Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.execution.rollout_ops import (
from ray.rllib.execution.train_ops import (
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.metrics import (
from ray.rllib.utils.typing import (
from ray.tune.logger import Logger
Sets the evaluation related configuration.
        Returns:
            This updated AlgorithmConfig object.
        