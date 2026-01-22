import gym
import json
import numpy as np
from copy import deepcopy
from minerl.herobraine.hero import mc, handlers
from collections import defaultdict, deque
def _get_start_velocity(self):
    if len(self.actions) < 2:
        return None
    a, a1 = (self.actions[0], self.actions[1])
    vx = a1['xpos'] - a['xpos']
    vy = a1['ypos'] - a['ypos']
    vz = a1['zpos'] - a['zpos']
    return (vx, vy, vz)