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
class LoopCraftingAgent:
    """
    Agent that acts according to the chain
    """

    def __init__(self, crafting_actions):
        """
        :param crafting_actions: list of crafting actions list({},...)
        """
        self.crafting_actions = crafting_actions
        self.current_action_index = 0

    def get_crafting_action(self):
        """
        :return: action to be taken
        """
        if len(self.crafting_actions) == 0:
            return {}
        result = self.crafting_actions[self.current_action_index]
        self.current_action_index = (self.current_action_index + 1) % len(self.crafting_actions)
        return result

    def reset_index(self):
        self.current_action_index = 0