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
class Food:

    def __init__(self, position=None):
        """
        Initializes the Food object with an optional position.

        Args:
            position (numpy array, optional): The initial position of the food. If not provided, a random position will be generated.
        """
        self.position = position if position is not None else self.generate_random_position()
        logging.debug(f'Food created at position {self.position}')

    def generate_random_position(self):
        """
        Generates a random position for the food within the grid boundaries.

        Returns:
            numpy array: The generated random position.
        """
        x = np.random.randint(0, WIDTH)
        y = np.random.randint(0, HEIGHT)
        position = np.array([x, y])
        logging.debug(f'Generated random food position: {position}')
        return position

    def draw(self, WIN):
        """
        Draws the food on the game window.

        Args:
            WIN (pygame.Surface): The game window surface to draw on.
        """
        x, y = self.position
        pygame.draw.rect(WIN, RED, (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        logging.debug(f'Food drawn at position {self.position}')