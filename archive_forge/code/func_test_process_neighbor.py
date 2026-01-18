import unittest
import math
from A_STAR import A_STAR
from pygame.math import Vector2
from typing import List, Tuple, Any, Set
from Utility import Grid, Node
from Snake import Snake
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
def test_process_neighbor(self):
    snake = Snake()
    current_node = Node(1, 1)
    neighbor = Node(1, 2)
    goalstate = Node(4, 4)
    self.a_star.process_neighbor(snake, current_node, neighbor, goalstate)
    self.assertEqual(neighbor.g, 2)
    self.assertEqual(neighbor.h, 6)
    self.assertEqual(neighbor.f, 8)