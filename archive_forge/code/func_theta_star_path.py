import pygame as pg
import sys
from random import randint, seed
from collections import deque
from typing import List, Tuple, Deque, Optional, Set, Dict
import logging
import math
from queue import PriorityQueue
def theta_star_path(self, start: Tuple[int, int], goal: Tuple[int, int], snake_body: Deque[Tuple[int, int]], grid: Grid) -> List[Tuple[int, int]]:
    """
        Theta* pathfinding algorithm to find the most efficient path from start to goal.
        This method extends A* by integrating line-of-sight checks to potentially skip nodes.

        Args:
            start (Tuple[int, int]): The starting position of the path.
            goal (Tuple[int, int]): The goal position of the path.
            snake_body (Deque[Tuple[int, int]]): The current positions occupied by the snake.
            grid (Grid): The grid on which the pathfinding operates.

        Returns:
            List[Tuple[int, int]]: The path from start to goal as a list of tuples.
        """

    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def line_of_sight(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """Check if a straight line between p1 and p2 is unobstructed."""
        x0, y0 = p1
        x1, y1 = p2
        dx, dy = (x1 - x0, y1 - y0)
        f = 0
        if dy < 0:
            dy = -dy
            sy = -1
        else:
            sy = 1
        if dx < 0:
            dx = -dx
            sx = -1
        else:
            sx = 1
        if dx >= dy:
            while x0 != x1:
                f += dy
                if f >= dx:
                    if grid.is_position_occupied((x0 + (sx - 1) // 2, y0 + (sy - 1) // 2)):
                        return False
                    y0 += sy
                    f -= dx
                if f != 0 and grid.is_position_occupied((x0 + (sx - 1) // 2, y0 + (sy - 1) // 2)):
                    return False
                if dy == 0 and grid.is_position_occupied((x0 + (sx - 1) // 2, y0)) and grid.is_position_occupied((x0 + (sx - 1) // 2, y0 - 1)):
                    return False
                x0 += sx
        else:
            while y0 != y1:
                f += dx
                if f >= dy:
                    if grid.is_position_occupied((x0 + (sx - 1) // 2, y0 + (sy - 1) // 2)):
                        return False
                    x0 += sx
                    f -= dy
                if f != 0 and grid.is_position_occupied((x0, y0 + (sy - 1) // 2)):
                    return False
                if dx == 0 and grid.is_position_occupied((x0, y0 + (sy - 1) // 2)) and grid.is_position_occupied((x0 - 1, y0 + (sy - 1) // 2)):
                    return False
                y0 += sy
        return True
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    while not open_set.empty():
        current = open_set.get()[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        for dx, dy in [(0, grid.block_size), (0, -grid.block_size), (grid.block_size, 0), (-grid.block_size, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < grid.width and 0 <= neighbor[1] < grid.height):
                continue
            if neighbor in snake_body:
                continue
            tentative_g_score = g_score[current] + grid.block_size
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if neighbor not in (n[1] for n in open_set.queue):
                    open_set.put((f_score[neighbor], neighbor))
                elif line_of_sight(came_from[current], neighbor):
                    came_from[neighbor] = came_from[current]
    return []