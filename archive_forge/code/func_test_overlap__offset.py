from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_overlap__offset(self):
    """Ensure an offset overlap intersection is correctly calculated."""
    mask1 = pygame.mask.Mask((65, 3), fill=True)
    mask2 = pygame.mask.Mask((66, 4), fill=True)
    mask1_count = mask1.count()
    mask2_count = mask2.count()
    mask1_size = mask1.get_size()
    mask2_size = mask2.get_size()
    for offset in self.ORIGIN_OFFSETS:
        msg = f'offset={offset}'
        expected_pos = (max(offset[0], 0), max(offset[1], 0))
        overlap_pos = mask1.overlap(other=mask2, offset=offset)
        self.assertEqual(overlap_pos, expected_pos, msg)
        self.assertEqual(mask1.count(), mask1_count, msg)
        self.assertEqual(mask2.count(), mask2_count, msg)
        self.assertEqual(mask1.get_size(), mask1_size, msg)
        self.assertEqual(mask2.get_size(), mask2_size, msg)