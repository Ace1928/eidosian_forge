import logging
import gymnasium as gym
import numpy as np
from typing import Callable, List, Optional, Tuple, Union, Set
from ray.rllib.env.base_env import BaseEnv, _DUMMY_AGENT_ID
from ray.rllib.utils.annotations import Deprecated, override, PublicAPI
from ray.rllib.utils.typing import (
from ray.util import log_once
Resets all or one particular sub-environment's state (by index).

        Args:
            idx: The index to reset at. If None, reset all the sub-environments' states.
        