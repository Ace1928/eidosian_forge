import gym
import json
import numpy as np
from copy import deepcopy
from minerl.herobraine.hero import mc, handlers
from collections import defaultdict, deque
def _patch_agent_start(self):
    old_create_agent_start = self.task.create_agent_start

    def create_agent_start():
        h = old_create_agent_start()
        start_pos = self._get_start_pos()
        start_velocity = self._get_start_velocity()
        if start_pos is not None:
            h.append(handlers.AgentStartPlacement(*start_pos))
        if start_velocity is not None:
            h.append(handlers.AgentStartVelocity(*start_velocity))
        return h
    self.task.create_agent_start = create_agent_start