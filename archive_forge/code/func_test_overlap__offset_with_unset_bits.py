from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_overlap__offset_with_unset_bits(self):
    """Ensure an offset overlap intersection is correctly calculated
        when (0, 0) bits not set."""
    mask1 = pygame.mask.Mask((65, 3), fill=True)
    mask2 = pygame.mask.Mask((66, 4), fill=True)
    unset_pos = (0, 0)
    mask1.set_at(unset_pos, 0)
    mask2.set_at(unset_pos, 0)
    mask1_count = mask1.count()
    mask2_count = mask2.count()
    mask1_size = mask1.get_size()
    mask2_size = mask2.get_size()
    for offset in self.ORIGIN_OFFSETS:
        msg = f'offset={offset}'
        x, y = offset
        expected_y = max(y, 0)
        if 0 == y:
            expected_x = max(x + 1, 1)
        elif 0 < y:
            expected_x = max(x + 1, 0)
        else:
            expected_x = max(x, 1)
        overlap_pos = mask1.overlap(mask2, Vector2(offset))
        self.assertEqual(overlap_pos, (expected_x, expected_y), msg)
        self.assertEqual(mask1.count(), mask1_count, msg)
        self.assertEqual(mask2.count(), mask2_count, msg)
        self.assertEqual(mask1.get_size(), mask1_size, msg)
        self.assertEqual(mask2.get_size(), mask2_size, msg)
        self.assertEqual(mask1.get_at(unset_pos), 0, msg)
        self.assertEqual(mask2.get_at(unset_pos), 0, msg)