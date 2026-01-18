from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_overlap_mask__offset_boundary(self):
    """Ensures overlap_mask handles offsets and boundaries correctly."""
    mask1 = pygame.mask.Mask((9, 3), fill=True)
    mask2 = pygame.mask.Mask((11, 5), fill=True)
    mask1_count = mask1.count()
    mask2_count = mask2.count()
    mask1_size = mask1.get_size()
    mask2_size = mask2.get_size()
    expected_count = 0
    expected_size = mask1_size
    offsets = ((mask1_size[0], 0), (0, mask1_size[1]), (-mask2_size[0], 0), (0, -mask2_size[1]))
    for offset in offsets:
        msg = f'offset={offset}'
        overlap_mask = mask1.overlap_mask(mask2, offset)
        self.assertIsInstance(overlap_mask, pygame.mask.Mask, msg)
        self.assertEqual(overlap_mask.count(), expected_count, msg)
        self.assertEqual(overlap_mask.get_size(), expected_size, msg)
        self.assertEqual(mask1.count(), mask1_count, msg)
        self.assertEqual(mask2.count(), mask2_count, msg)
        self.assertEqual(mask1.get_size(), mask1_size, msg)
        self.assertEqual(mask2.get_size(), mask2_size, msg)