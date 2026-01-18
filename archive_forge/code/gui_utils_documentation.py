from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame

    Updates the GUI with the current game state.

    Args:
        board (np.ndarray): The game board as a 2D NumPy array.
        score (int): The current score.
    