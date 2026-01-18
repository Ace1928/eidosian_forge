from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_scale(self):
    sizes = ((100, 0), (0, 100), (0, 0))
    for size in sizes:
        mask = pygame.mask.Mask(size)
        mask2 = mask.scale((2, 3))
        self.assertIsInstance(mask2, pygame.mask.Mask)
        self.assertEqual(mask2.get_size(), (2, 3))