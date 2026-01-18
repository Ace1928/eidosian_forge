from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_set_at__out_of_bounds(self):
    """Ensure set_at() checks bounds."""
    width, height = (11, 3)
    mask = pygame.mask.Mask((width, height))
    with self.assertRaises(IndexError):
        mask.set_at((width, 0))
    with self.assertRaises(IndexError):
        mask.set_at((0, height))
    with self.assertRaises(IndexError):
        mask.set_at((-1, 0))
    with self.assertRaises(IndexError):
        mask.set_at((0, -1))