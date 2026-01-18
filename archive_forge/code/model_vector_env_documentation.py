import logging
from gymnasium.spaces import Discrete
import numpy as np
from ray.rllib.utils.annotations import override
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.evaluation.rollout_worker import get_global_worker
from ray.rllib.env.base_env import BaseEnv, convert_to_base_env
from ray.rllib.utils.typing import EnvType
Override parent to store actual env obs for upcoming predictions.