import gym
import json
import numpy as np
from copy import deepcopy
from minerl.herobraine.hero import mc, handlers
from collections import defaultdict, deque

        Adjusts stats (currently, only inventory) by the amount at the end of the replay
        