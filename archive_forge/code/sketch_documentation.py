import snake
import apple
import search
import logging
from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint

    The main function orchestrates the entire game by integrating components from snake.py, apple.py, and search.py.
    It initializes the game environment, processes user inputs, updates game states, and renders the game elements.
    