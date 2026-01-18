import pygame
import random
import heapq
import logging
from typing import List, Optional, Dict, Any, Tuple
import cProfile
from collections import deque
import numpy as np
import time
import torch
from functools import lru_cache as LRUCache
import math
import asyncio
from scipy.spatial import Delaunay
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from queue import PriorityQueue
from collections import defaultdict
def update_position(self, position, value):
    """
        Updates the grid cell at a given position with a specified value.

        Args:
            position (numpy array): The position to update on the grid.
            value (int): The value to set at the given position.
        """
    x, y = position
    self.cells[y, x] = value
    logging.debug(f'Grid position {position} updated with value {value}')