from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_get_bounding_rects(self):
    """Ensures get_bounding_rects correctly handles zero sized masks."""
    expected_bounding_rects = []
    for size in ((21, 0), (0, 20), (0, 0)):
        mask = pygame.mask.Mask(size)
        bounding_rects = mask.get_bounding_rects()
        self.assertListEqual(bounding_rects, expected_bounding_rects, f'size={size}')