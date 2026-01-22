import math
import unittest
import sys
import warnings
import pygame
from pygame import draw
from pygame import draw_py
from pygame.locals import SRCALPHA
from pygame.tests import test_utils
from pygame.math import Vector2
class DrawAALinesTest(AALinesMixin, DrawTestCase):
    """Test draw module function aalines.

    This class inherits the general tests from AALinesMixin. It is also the
    class to add any draw.aalines specific tests to.
    """