from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.core.learner.learner import (
from ray.rllib.core.rl_module.rl_module import ModuleID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.lambda_defaultdict import LambdaDefaultDict
from ray.rllib.utils.metrics import (
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.typing import ResultDict
@dataclass
class ImpalaLearnerHyperparameters(LearnerHyperparameters):
    """LearnerHyperparameters for the ImpalaLearner sub-classes (framework specific).

    These should never be set directly by the user. Instead, use the IMPALAConfig
    class to configure your algorithm.
    See `ray.rllib.algorithms.impala.impala::IMPALAConfig::training()` for more details
    on the individual properties.

    Attributes:
        rollout_frag_or_episode_len: The length of a rollout fragment or episode.
            Used when making SampleBatches time major for computing loss.
        recurrent_seq_len: The length of a recurrent sequence. Used when making
            SampleBatches time major for computing loss.
    """
    rollout_frag_or_episode_len: int = None
    recurrent_seq_len: int = None
    discount_factor: float = None
    vtrace_clip_rho_threshold: float = None
    vtrace_clip_pg_rho_threshold: float = None
    vf_loss_coeff: float = None
    entropy_coeff: float = None
    entropy_coeff_schedule: Optional[List[List[Union[int, float]]]] = None