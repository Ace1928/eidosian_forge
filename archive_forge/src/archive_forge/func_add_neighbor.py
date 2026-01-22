import pygame
import pygame_gui
import numpy as np
from collections import deque
from typing import List, Tuple, Deque, Dict, Any, Optional
import threading
import time
import random
import math
import asyncio
import os
import logging
import sys
import aiofiles
from functools import lru_cache as LRUCache
import aiohttp
import json
import cachetools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.distributed as dist
import torch.nn.parallel as parallel
import torch.utils.data.distributed as distributed
import torch.distributions as distributions
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
import torch.cuda as cuda  # Added for potential GPU acceleration
import torch.backends.cudnn as cudnn  # Added for optimizing deep learning computations on CUDA
import logging  # For detailed logging of operations and errors
import hashlib  # For generating unique identifiers for nodes
import bisect  # For maintaining sorted lists
import gc  # For explicit garbage collection if necessary
@staticmethod
def add_neighbor(self, neighbor: 'Node'):
    """
        Add a neighboring node to this node's list of neighbors using a cache mechanism to ensure efficient memory usage.

        Args:
            neighbor (Node): The neighbor node to add.
        """
    if neighbor.id not in self.neighbors:
        self.neighbors[neighbor.id] = neighbor
    logging.debug(f'Added neighbor {neighbor} to node at {self.position.tolist()} with efficient caching mechanism')