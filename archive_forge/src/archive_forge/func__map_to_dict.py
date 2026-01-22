import os
from collections import defaultdict
import numpy as np
import minerl
import time
from typing import List
import collections
import os
import cv2
import gym
from minerl.data import DataPipeline
import sys
import time
from collections import deque, defaultdict
from enum import Enum
def _map_to_dict(i: int, src: list, key: str, gym_space: gym.spaces.space, dst: dict):
    if isinstance(gym_space, gym.spaces.Dict):
        dont_count = False
        inner_dict = collections.OrderedDict()
        for idx, (k, s) in enumerate(gym_space.spaces.items()):
            if key in ['equipped_items', 'mainhand']:
                dont_count = True
                i = _map_to_dict(i, src, k, s, inner_dict)
            else:
                _map_to_dict(idx, src[i].T, k, s, inner_dict)
        dst[key] = inner_dict
        if dont_count:
            return i
        else:
            return i + 1
    else:
        dst[key] = src[i]
        return i + 1