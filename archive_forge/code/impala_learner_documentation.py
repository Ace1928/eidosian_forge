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
Reduce/Aggregate a list of results from Impala Learners.

    Average the values of the result dicts. Add keys for the number of agent and env
    steps trained (on all modules).

    Args:
        results: result dicts to reduce.

    Returns:
        A reduced result dict.
    