import dataclasses
from typing import Optional, Type
import logging
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.appo.appo_learner import (
from ray.rllib.algorithms.impala.impala import Impala, ImpalaConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics import ALL_MODULES, LEARNER_STATS_KEY
from ray.rllib.utils.typing import (
Updates the target network and the KL coefficient for the APPO-loss.

        This method is called from within the `training_step` method after each train
        update.
        The target network update frequency is calculated automatically by the product
        of `num_sgd_iter` setting (usually 1 for APPO) and `minibatch_buffer_size`.

        Args:
            train_results: The results dict collected during the most recent
                training step.
        