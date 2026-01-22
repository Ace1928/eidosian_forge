import numpy as np
from typing import Optional
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING
class RandomAgentConfig(AlgorithmConfig):
    """Defines a configuration class from which a RandomAgent Algorithm can be built.

    Example:
        >>> from ray.rllib.algorithms.random_agent import RandomAgentConfig
        >>> config = RandomAgentConfig().rollouts(rollouts_per_iteration=20)
        >>> print(config.to_dict()) # doctest: +SKIP
        >>> # Build an Algorithm object from the config and run 1 training iteration.
        >>> algo = config.build(env="CartPole-v1")
        >>> algo.train() # doctest: +SKIP
    """

    def __init__(self, algo_class=None):
        """Initializes a RandomAgentConfig instance."""
        super().__init__(algo_class=algo_class or RandomAgent)
        self.rollouts_per_iteration = 10

    def rollouts(self, *, rollouts_per_iteration: Optional[int]=NotProvided, **kwargs) -> 'RandomAgentConfig':
        """Sets the rollout configuration.

        Args:
            rollouts_per_iteration: How many episodes to run per training iteration.

        Returns:
            This updated AlgorithmConfig object.
        """
        super().rollouts(**kwargs)
        if rollouts_per_iteration is not NotProvided:
            self.rollouts_per_iteration = rollouts_per_iteration
        return self