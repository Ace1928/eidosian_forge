import gym
import json
import numpy as np
from copy import deepcopy
from minerl.herobraine.hero import mc, handlers
from collections import defaultdict, deque
def _get_start_pos(self):
    if len(self.actions) == 0:
        return None
    a = self.actions[0]
    return (a['xpos'], a['ypos'], a['zpos'], a['yaw'], a['pitch'])