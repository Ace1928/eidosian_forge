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

        Returns a generator for iterating through batches of the dataset.
        :param batch_size:
        :param number_of_workers:
        :param worker_batch_size:
        :param size_to_dequeue:
        :return:
        