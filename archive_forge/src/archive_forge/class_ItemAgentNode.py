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
class ItemAgentNode:
    """
    combined info about each agent
    """

    def __init__(self, node_name, count_, pov_agent, crafting_agent):
        self.name = node_name
        self.count = count_
        self.pov_agent = pov_agent
        self.crafting_agent = crafting_agent
        self.success = deque([0], maxlen=10)
        self.eps_to_save = 0
        self.model_dir = 'train/' + self.name
        self.exploration_force = True
        self.fixed = False

    def load_agent(self, load_dir=None):
        if load_dir is None:
            load_dir = self.model_dir
        self.pov_agent.load_agent(load_dir)