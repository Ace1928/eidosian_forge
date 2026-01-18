import random
import string
import gym
import gym.spaces
import numpy as np
import random
from collections import OrderedDict
from typing import List
import gym
import logging
import gym.spaces
import numpy as np
import collections
import warnings
import abc
def unflattenable_map(self, x: OrderedDict) -> OrderedDict:
    """
        Selects the unflattened part of x
        """
    return OrderedDict({k: v.unflattenable_map(x[k]) if hasattr(v, 'unflattenable_map') else x[k] for k, v in self.spaces.items() if not v.is_flattenable()})