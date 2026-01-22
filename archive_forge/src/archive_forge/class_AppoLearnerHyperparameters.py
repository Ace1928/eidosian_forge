import abc
from dataclasses import dataclass
from typing import Any, Mapping
from ray.rllib.algorithms.impala.impala_learner import (
from ray.rllib.core.rl_module.marl_module import ModuleID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.lambda_defaultdict import LambdaDefaultDict
from ray.rllib.utils.metrics import LAST_TARGET_UPDATE_TS, NUM_TARGET_UPDATES
from ray.rllib.utils.schedules.scheduler import Scheduler
@dataclass
class AppoLearnerHyperparameters(ImpalaLearnerHyperparameters):
    """Hyperparameters for the APPOLearner sub-classes (framework specific).

    These should never be set directly by the user. Instead, use the APPOConfig
    class to configure your algorithm.
    See `ray.rllib.algorithms.appo.appo::APPOConfig::training()` for more details on the
    individual properties.
    """
    use_kl_loss: bool = None
    kl_coeff: float = None
    kl_target: float = None
    clip_param: float = None
    tau: float = None
    target_update_frequency_ts: int = None