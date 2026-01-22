from copy import deepcopy
from gymnasium.spaces import (
import numpy as np
from typing import Any, Dict, Optional, Tuple
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, AgentID
def _convert_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw observations

        These conversions are necessary to make the observations fall into the
        observation space defined below.
        """
    new_obs = deepcopy(obs)
    if new_obs['players_raw'][0]['ball_owned_team'] == -1:
        new_obs['players_raw'][0]['ball_owned_team'] = 2
    if new_obs['players_raw'][0]['ball_owned_player'] == -1:
        new_obs['players_raw'][0]['ball_owned_player'] = 11
    new_obs['players_raw'][0]['steps_left'] = [new_obs['players_raw'][0]['steps_left']]
    return new_obs