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
@property
def unflattened(self):
    """
        Returns the unflatteneable part of the space.
        """
    if not hasattr(self, '_unflattened'):
        self._unflattened = self.create_unflattened_space()
    return self._unflattened