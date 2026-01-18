from copy import deepcopy
from gymnasium.spaces import (
import numpy as np
from typing import Any, Dict, Optional, Tuple
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, AgentID
Construct the action and observation spaces

        Description of actions and observations:
        https://github.com/google-research/football/blob/master/gfootball/doc/
        observation.md
        