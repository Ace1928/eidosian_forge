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
def test_overlap_area__offset(self):
    """Ensure an offset overlap_area is correctly calculated."""
    mask1 = pygame.mask.Mask((65, 3), fill=True)
    mask2 = pygame.mask.Mask((66, 4), fill=True)
    mask1_count = mask1.count()
    mask2_count = mask2.count()
    mask1_size = mask1.get_size()
    mask2_size = mask2.get_size()
    rect1 = mask1.get_rect()
    rect2 = mask2.get_rect()
    for offset in self.ORIGIN_OFFSETS:
        msg = f'offset={offset}'
        rect2.topleft = offset
        overlap_rect = rect1.clip(rect2)
        expected_count = overlap_rect.w * overlap_rect.h
        overlap_count = mask1.overlap_area(other=mask2, offset=offset)
        self.assertEqual(overlap_count, expected_count, msg)
        self.assertEqual(mask1.count(), mask1_count, msg)
        self.assertEqual(mask2.count(), mask2_count, msg)
        self.assertEqual(mask1.get_size(), mask1_size, msg)
        self.assertEqual(mask2.get_size(), mask2_size, msg)