import pygame as pg
import sys
from random import randint, seed
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Deque, Set, Optional
from heapq import heappush, heappop
import numpy as np
import math
from queue import PriorityQueue
import logging
def _a_star_search_helper(self, start: Tuple[int, int], goal: Tuple[int, int], is_return_path: bool=False) -> List[Tuple[int, int]]:
    """
        Helper function to perform the A* search algorithm for a single path.
        This implementation is optimized for real-time performance and scalability, using
        efficient data structures and pruning techniques to minimize the search space.
        It also incorporates adaptive heuristics and dynamic adjustments based on the current
        game state to generate strategic and efficient paths.

        Args:
            start (Tuple[int, int]): The starting position of the path.
            goal (Tuple[int, int]): The goal position of the path.
            is_return_path (bool): Flag indicating if the search is for the return path.

        Returns:
            List[Tuple[int, int]]: The optimal path from start to goal as a list of coordinates.
        """
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {start: None}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    f_score = defaultdict(lambda: float('inf'))
    f_score[start] = self.heuristic(start, goal, is_return_path=is_return_path)
    closed_set = set()
    last_direction = None
    while not open_set.empty():
        current = open_set.get()[1]
        closed_set.add(current)
        if current == goal:
            return self.reconstruct_path(came_from, current)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < BLOCK_SIZE and 0 <= neighbor[1] < BLOCK_SIZE and (neighbor not in self.snake) and (neighbor not in self.obstacles) and (neighbor not in closed_set):
                tentative_g_score = g_score[current] + math.sqrt(dx ** 2 + dy ** 2)
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal, (dx, dy), is_return_path)
                    open_set.put((f_score[neighbor], neighbor))
                    last_direction = (dx, dy)
        if open_set.qsize() > OPEN_SET_LIMIT:
            pruned_open_set = PriorityQueue()
            for _ in range(OPEN_SET_LIMIT):
                pruned_open_set.put(open_set.get())
            open_set = pruned_open_set
    return []