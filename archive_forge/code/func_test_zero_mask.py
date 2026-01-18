from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask(self):
    """Ensures masks can be created with zero sizes."""
    for size in ((100, 0), (0, 100), (0, 0)):
        for fill in (True, False):
            msg = f'size={size}, fill={fill}'
            mask = pygame.mask.Mask(size, fill=fill)
            self.assertIsInstance(mask, pygame.mask.Mask, msg)
            self.assertEqual(mask.get_size(), size, msg)