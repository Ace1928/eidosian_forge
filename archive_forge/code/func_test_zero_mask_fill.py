from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_fill(self):
    """Ensures fill correctly handles zero sized masks."""
    expected_count = 0
    for size in ((100, 0), (0, 100), (0, 0)):
        mask = pygame.mask.Mask(size)
        mask.fill()
        self.assertEqual(mask.count(), expected_count, f'size={size}')