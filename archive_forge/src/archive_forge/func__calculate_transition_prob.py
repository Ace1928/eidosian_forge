from contextlib import closing
from io import StringIO
from os import path
from typing import Optional
import numpy as np
from gym import Env, logger, spaces
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled
def _calculate_transition_prob(self, current, delta):
    """Determine the outcome for an action. Transition Prob is always 1.0.

        Args:
            current: Current position on the grid as (row, col)
            delta: Change in position for transition

        Returns:
            Tuple of ``(1.0, new_state, reward, terminated)``
        """
    new_position = np.array(current) + np.array(delta)
    new_position = self._limit_coordinates(new_position).astype(int)
    new_state = np.ravel_multi_index(tuple(new_position), self.shape)
    if self._cliff[tuple(new_position)]:
        return [(1.0, self.start_state_index, -100, False)]
    terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
    is_terminated = tuple(new_position) == terminal_state
    return [(1.0, new_state, -1, is_terminated)]