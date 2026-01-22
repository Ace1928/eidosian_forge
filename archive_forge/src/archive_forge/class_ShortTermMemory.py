import numpy as np  # Assuming NumPy is used for efficient array manipulation
import random
import types
import importlib.util
import logging
import collections
from typing import Deque, Dict, Tuple, List
from typing import List, Tuple
from functools import wraps
import logging
class ShortTermMemory:
    """
    Manages the short-term memory for the AI, storing recent moves and their outcomes.
    """

    @StandardDecorator()
    def __init__(self, capacity: int=10):
        self.memory: Deque[Tuple[np.ndarray, str, int]] = collections.deque(maxlen=capacity)

    @StandardDecorator
    def store(self, board: np.ndarray, move: str, score: int) -> None:
        """
        Stores the given game state, move, and score in the short-term memory.

        Args:
            board (np.ndarray): The current game board.
            move (str): The move made.
            score (int): The score gained from the move.
        """
        self.memory.append((board, move, score))