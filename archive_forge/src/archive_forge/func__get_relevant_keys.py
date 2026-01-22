from collections import deque
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import gym.error
from gym import Env, logger
from gym.core import ActType, ObsType
from gym.error import DependencyNotInstalled
from gym.logger import deprecation
def _get_relevant_keys(self, keys_to_action: Optional[Dict[Tuple[int], int]]=None) -> set:
    if keys_to_action is None:
        if hasattr(self.env, 'get_keys_to_action'):
            keys_to_action = self.env.get_keys_to_action()
        elif hasattr(self.env.unwrapped, 'get_keys_to_action'):
            keys_to_action = self.env.unwrapped.get_keys_to_action()
        else:
            raise MissingKeysToAction(f'{self.env.spec.id} does not have explicit key to action mapping, please specify one manually')
    assert isinstance(keys_to_action, dict)
    relevant_keys = set(sum((list(k) for k in keys_to_action.keys()), []))
    return relevant_keys