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
class Pathfinding:

    def __init__(self, grid):
        """
        Initialize the Pathfinding class which is responsible for calculating optimal paths for the snake using advanced algorithms.

        Args:
            grid (Grid): The game grid which contains all the necessary information about the game state.

        Attributes:
            grid (Grid): Stores the reference to the game grid.
            path_cache (LRUCache): Caches the results of complex path calculations for quick retrieval.
            pathfinding_lock (asyncio.Lock): An asyncio lock to ensure thread-safe operations during path calculations.
        """
        self.grid = grid
        self.path_cache = LRUCache(maxsize=1024)
        self.pathfinding_lock = asyncio.Lock()

    async def calculate_path(self, start, goal):
        """
        Asynchronously calculates the optimal path from start to goal using A* algorithm.

        Args:
            start (Node): The starting node of the path.
            goal (Node): The goal node of the path.

        Returns:
            List[Node]: The optimal path as a list of nodes.
        """
        async with self.pathfinding_lock:
            path_key = (hash(start), hash(goal))
            if path_key in self.path_cache:
                return self.path_cache[path_key]
            path = self._a_star_search(start, goal)
            self.path_cache[path_key] = path
            return path

    def _a_star_search(self, start, goal):
        """
        Implements the A* search algorithm to find the shortest path from start to goal.

        Args:
            start (Node): The starting node.
            goal (Node): The goal node.

        Returns:
            List[Node]: The path from start to goal as a list of nodes.
        """
        open_set = deque([start])
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.grid._heuristic(start, goal)}
        while open_set:
            current = min(open_set, key=lambda o: f_score[o])
            if current == goal:
                return self.grid._reconstruct_path(came_from, current)
            open_set.remove(current)
            for neighbor in self.grid.get_neighbors(current):
                tentative_g_score = g_score[current] + self.grid.distance(current, neighbor)
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.grid._heuristic(neighbor, goal)
                    if neighbor not in open_set:
                        open_set.append(neighbor)
        return []

    @staticmethod
    def _heuristic(node1, node2):
        """
        Calculate the heuristic estimated cost from node1 to node2 using Manhattan distance.

        Args:
            node1 (Node): The first node.
            node2 (Node): The second node.

        Returns:
            float: The estimated cost from node1 to node2.
        """
        return np.linalg.norm(np.array(node1.position) - np.array(node2.position))

    @staticmethod
    def _reconstruct_path(came_from, current):
        """
        Reconstruct the path from start to goal using the came_from map.

        Args:
            came_from (dict): The map of nodes to their predecessors.
            current (Node): The current node to start reconstruction from.

        Returns:
            List[Node]: The reconstructed path as a list of nodes.
        """
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]