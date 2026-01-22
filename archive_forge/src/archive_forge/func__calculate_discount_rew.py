import json
import logging
import multiprocessing
import os
from collections import OrderedDict
from queue import Queue, PriorityQueue
from typing import List, Tuple, Any
import cv2
import numpy as np
from multiprocess.pool import Pool
from minerl.herobraine.hero.agent_handler import HandlerCollection, AgentHandler
from minerl.herobraine.hero.handlers import RewardHandler
@staticmethod
def _calculate_discount_rew(rewards, gamma):
    total_reward = 0
    for i, rew in enumerate(rewards):
        total_reward += gamma ** i * rew
    return total_reward