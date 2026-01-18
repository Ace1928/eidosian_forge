from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_convolve(self):
    """Ensures convolve correctly handles zero sized masks.

        Tests the different combinations of sized and zero sized masks.
        """
    for size1 in ((17, 13), (71, 0), (0, 70), (0, 0)):
        mask1 = pygame.mask.Mask(size1, fill=True)
        for size2 in ((11, 7), (81, 0), (0, 60), (0, 0)):
            msg = f'sizes={size1}, {size2}'
            mask2 = pygame.mask.Mask(size2, fill=True)
            expected_size = (max(0, size1[0] + size2[0] - 1), max(0, size1[1] + size2[1] - 1))
            mask = mask1.convolve(mask2)
            self.assertIsInstance(mask, pygame.mask.Mask, msg)
            self.assertIsNot(mask, mask2, msg)
            self.assertEqual(mask.get_size(), expected_size, msg)