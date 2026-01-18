from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
@unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
def test_overlap_area(self):
    """Ensure the overlap_area is correctly calculated.

        Testing the different combinations of full/empty masks:
            (mask1-filled) 1 overlap_area 1 (mask2-filled)
            (mask1-empty)  0 overlap_area 1 (mask2-filled)
            (mask1-filled) 1 overlap_area 0 (mask2-empty)
            (mask1-empty)  0 overlap_area 0 (mask2-empty)
        """
    expected_size = width, height = (4, 4)
    offset = (0, 0)
    expected_default = 0
    expected_counts = {(True, True): width * height}
    for fill2 in (True, False):
        mask2 = pygame.mask.Mask(expected_size, fill=fill2)
        mask2_count = mask2.count()
        for fill1 in (True, False):
            key = (fill1, fill2)
            msg = f'key={key}'
            mask1 = pygame.mask.Mask(expected_size, fill=fill1)
            mask1_count = mask1.count()
            expected_count = expected_counts.get(key, expected_default)
            overlap_count = mask1.overlap_area(mask2, offset)
            self.assertEqual(overlap_count, expected_count, msg)
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), expected_size, msg)
            self.assertEqual(mask2.get_size(), expected_size, msg)