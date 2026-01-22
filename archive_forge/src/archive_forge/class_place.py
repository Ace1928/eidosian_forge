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
class place(Enum):
    cobblestone = 0
    crafting_table = 1
    dirt = 2
    furnace = 3
    none = 4
    stone = 5
    torch = 6