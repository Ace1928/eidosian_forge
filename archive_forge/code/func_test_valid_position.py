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
def test_valid_position(self):
    node = Node(1, 1)
    self.assertTrue(self.a_star.valid_position(node))