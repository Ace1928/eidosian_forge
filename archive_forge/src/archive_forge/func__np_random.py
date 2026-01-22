import sys
from typing import (
import numpy as np
from gym import spaces
from gym.logger import warn
from gym.utils import seeding
@property
def _np_random(self):
    raise AttributeError("Can't access `_np_random` of a wrapper, use `.unwrapped._np_random` or `.np_random`.")