import gym
import json
import numpy as np
from copy import deepcopy
from minerl.herobraine.hero import mc, handlers
from collections import defaultdict, deque
def update_stats(self, ob, info):
    replaying = info[ReplayWrapper.IGNORE_POLICY_ACTION]
    if replaying:
        self.last_info = deepcopy(info)
        self.last_ob = deepcopy(ob)
    if self.multiagent:
        info['agent_0'][ReplayWrapper.IGNORE_POLICY_ACTION] = replaying