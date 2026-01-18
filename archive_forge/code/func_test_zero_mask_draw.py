from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_draw(self):
    """Ensures draw correctly handles zero sized masks.

        Tests combinations of sized and zero sized masks.
        """
    offset = (0, 0)
    for size1, size2 in zero_size_pairs(31, 37):
        msg = f'size1={size1}, size2={size2}'
        mask1 = pygame.mask.Mask(size1, fill=True)
        mask2 = pygame.mask.Mask(size2, fill=True)
        expected_count = mask1.count()
        mask1.draw(mask2, offset)
        self.assertEqual(mask1.count(), expected_count, msg)
        self.assertEqual(mask1.get_size(), size1, msg)