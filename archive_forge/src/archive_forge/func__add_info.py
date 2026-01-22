from typing import Any, List, Optional, Tuple, Union
import numpy as np
import gym
from gym.vector.utils.spaces import batch_space
def _add_info(self, infos: dict, info: dict, env_num: int) -> dict:
    """Add env info to the info dictionary of the vectorized environment.

        Given the `info` of a single environment add it to the `infos` dictionary
        which represents all the infos of the vectorized environment.
        Every `key` of `info` is paired with a boolean mask `_key` representing
        whether or not the i-indexed environment has this `info`.

        Args:
            infos (dict): the infos of the vectorized environment
            info (dict): the info coming from the single environment
            env_num (int): the index of the single environment

        Returns:
            infos (dict): the (updated) infos of the vectorized environment

        """
    for k in info.keys():
        if k not in infos:
            info_array, array_mask = self._init_info_arrays(type(info[k]))
        else:
            info_array, array_mask = (infos[k], infos[f'_{k}'])
        info_array[env_num], array_mask[env_num] = (info[k], True)
        infos[k], infos[f'_{k}'] = (info_array, array_mask)
    return infos