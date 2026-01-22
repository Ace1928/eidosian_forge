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
def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation, info = env.reset(**data)
                write_to_shared_memory(observation_space, index, observation, shared_memory)
                pipe.send(((None, info), True))
            elif command == 'step':
                observation, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    old_observation, old_info = (observation, info)
                    observation, info = env.reset()
                    info['final_observation'] = old_observation
                    info['final_info'] = old_info
                write_to_shared_memory(observation_space, index, observation, shared_memory)
                pipe.send(((None, reward, terminated, truncated, info), True))
            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == '_call':
                name, args, kwargs = data
                if name in ['reset', 'step', 'seed', 'close']:
                    raise ValueError(f'Trying to call function `{name}` with `_call`. Use `{name}` directly instead.')
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == '_setattr':
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == '_check_spaces':
                pipe.send(((data[0] == observation_space, data[1] == env.action_space), True))
            else:
                raise RuntimeError(f'Received unknown command `{command}`. Must be one of {{`reset`, `step`, `seed`, `close`, `_call`, `_setattr`, `_check_spaces`}}.')
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()