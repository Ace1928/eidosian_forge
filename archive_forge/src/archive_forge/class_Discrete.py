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
class Discrete(gym.spaces.Discrete, MineRLSpace):

    def __init__(self, *args, **kwargs):
        super(Discrete, self).__init__(*args, **kwargs)
        self.eye = np.eye(self.n, dtype=np.float32)

    def no_op(self, batch_shape=()):
        if len(batch_shape) == 0:
            return 0
        else:
            return np.zeros(batch_shape).astype(self.dtype)

    def create_flattened_space(self):
        return Box(low=0, high=1, shape=(self.n,))

    def flat_map(self, x):
        return self.eye[x]

    def unmap(self, x):
        return np.array(np.argmax(x, axis=-1), dtype=self.dtype)

    def sample(self, bs=None):
        bdim = () if bs is None else (bs,)
        return self.np_random.randint(self.n, size=bdim)