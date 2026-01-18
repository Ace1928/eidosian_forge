from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_overlap__no_overlap(self):
    """Ensure an offset overlap intersection is correctly calculated
        when there is no overlap."""
    mask1 = pygame.mask.Mask((65, 3), fill=True)
    mask1_count = mask1.count()
    mask1_size = mask1.get_size()
    mask2_w, mask2_h = (67, 5)
    mask2_size = (mask2_w, mask2_h)
    mask2 = pygame.mask.Mask(mask2_size)
    set_pos = (mask2_w - 1, mask2_h - 1)
    mask2.set_at(set_pos)
    mask2_count = 1
    for offset in self.ORIGIN_OFFSETS:
        msg = f'offset={offset}'
        overlap_pos = mask1.overlap(mask2, offset)
        self.assertIsNone(overlap_pos, msg)
        self.assertEqual(mask1.count(), mask1_count, msg)
        self.assertEqual(mask2.count(), mask2_count, msg)
        self.assertEqual(mask1.get_size(), mask1_size, msg)
        self.assertEqual(mask2.get_size(), mask2_size, msg)
        self.assertEqual(mask2.get_at(set_pos), 1, msg)