import multiprocessing as mp
import sys
import time
from copy import deepcopy
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import gym
from gym import logger
from gym.core import ObsType
from gym.error import (
from gym.vector.utils import (
from gym.vector.vector_env import VectorEnv
def _raise_if_errors(self, successes):
    if all(successes):
        return
    num_errors = self.num_envs - sum(successes)
    assert num_errors > 0
    for i in range(num_errors):
        index, exctype, value = self.error_queue.get()
        logger.error(f'Received the following error from Worker-{index}: {exctype.__name__}: {value}')
        logger.error(f'Shutting down Worker-{index}.')
        self.parent_pipes[index].close()
        self.parent_pipes[index] = None
        if i == num_errors - 1:
            logger.error('Raising the last exception back to the main process.')
            raise exctype(value)