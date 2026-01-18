import gymnasium as gym
from gymnasium.spaces import Tuple, Dict
import numpy as np
from ray.rllib.utils.annotations import DeveloperAPI
import tree  # pip install dm_tree
from typing import Any, List, Optional, Union
@DeveloperAPI
def unsquash_action(action, action_space_struct):
    """Unsquashes all components in `action` according to the given Space.

    Inverse of `normalize_action()`. Useful for mapping policy action
    outputs (normalized between -1.0 and 1.0) to an env's action space.
    Unsquashing results in cont. action component values between the
    given Space's bounds (`low` and `high`). This only applies to Box
    components within the action space, whose dtype is float32 or float64.

    Args:
        action: The action to be unsquashed. This could be any complex
            action, e.g. a dict or tuple.
        action_space_struct: The action space struct,
            e.g. `{"a": Box()}` for a space: Dict({"a": Box()}).

    Returns:
        Any: The input action, but unsquashed, according to the space's
            bounds. An unsquashed action is ready to be sent to the
            environment (`BaseEnv.send_actions([unsquashed actions])`).
    """

    def map_(a, s):
        if isinstance(s, gym.spaces.Box) and np.all(s.bounded_below) and np.all(s.bounded_above):
            if s.dtype == np.float32 or s.dtype == np.float64:
                a = s.low + (a + 1.0) * (s.high - s.low) / 2.0
                a = np.clip(a, s.low, s.high)
            elif np.issubdtype(s.dtype, np.integer):
                a = s.low + a
        return a
    return tree.map_structure(map_, action, action_space_struct)