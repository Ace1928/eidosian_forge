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

        Reconstructs the path from the start to the end position using the came_from map generated during the pathfinding process.
        This method traces back from the end position to the start, efficiently compiling the sequence of steps taken to reach the goal.

        Args:
            came_from (dict): A dictionary mapping each position to its predecessor along the path.
            current (tuple): The endpoint of the path.

        Returns:
            list: The reconstructed path as a list of positions, starting from the initial position and ending at the goal.
        