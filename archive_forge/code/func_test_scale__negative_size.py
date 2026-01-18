from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_scale__negative_size(self):
    """Ensure scale handles negative sizes correctly."""
    mask = pygame.Mask((100, 100))
    with self.assertRaises(ValueError):
        mask.scale((-1, -1))
    with self.assertRaises(ValueError):
        mask.scale(Vector2(-1, 10))
    with self.assertRaises(ValueError):
        mask.scale((10, -1))