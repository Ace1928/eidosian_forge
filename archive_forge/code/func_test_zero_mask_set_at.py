from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_set_at(self):
    """Ensures set_at correctly handles zero sized masks."""
    for size in ((31, 0), (0, 30), (0, 0)):
        mask = pygame.mask.Mask(size)
        with self.assertRaises(IndexError):
            mask.set_at((0, 0))