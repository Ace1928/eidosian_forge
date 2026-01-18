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
@staticmethod
def to_old_chain_format(items: List[Item], return_time_indexes: bool):
    result = []
    used_actions = defaultdict(int)
    for item in items:
        for action in item.actions:
            full_action = f'{action.name}{action.value}'
            result.append(f'{action.name}:{used_actions[full_action]}:{action.value}')
            used_actions[full_action] += 1
        result.append(f'{item.name}:{item.value}')
    time_indexes = [(f'{item.name}+{item.value}', item.begin, item.end) for item in items]
    if return_time_indexes:
        return (result, time_indexes)
    return result