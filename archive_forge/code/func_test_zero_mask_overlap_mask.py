from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_overlap_mask(self):
    """Ensures overlap_mask correctly handles zero sized masks.

        Tests combinations of sized and zero sized masks.
        """
    offset = (0, 0)
    expected_count = 0
    for size1, size2 in zero_size_pairs(43, 53):
        msg = f'size1={size1}, size2={size2}'
        mask1 = pygame.mask.Mask(size1, fill=True)
        mask2 = pygame.mask.Mask(size2, fill=True)
        overlap_mask = mask1.overlap_mask(mask2, offset)
        self.assertIsInstance(overlap_mask, pygame.mask.Mask, msg)
        self.assertEqual(overlap_mask.count(), expected_count, msg)
        self.assertEqual(overlap_mask.get_size(), size1, msg)