from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_convolve__with_output_mask(self):
    """Ensures convolve correctly handles zero sized masks
        when using an output mask argument.

        Tests the different combinations of sized and zero sized masks.
        """
    for size1 in ((11, 17), (91, 0), (0, 90), (0, 0)):
        mask1 = pygame.mask.Mask(size1, fill=True)
        for size2 in ((13, 11), (83, 0), (0, 62), (0, 0)):
            mask2 = pygame.mask.Mask(size2, fill=True)
            for output_size in ((7, 5), (71, 0), (0, 70), (0, 0)):
                msg = f'sizes={size1}, {size2}, {output_size}'
                output_mask = pygame.mask.Mask(output_size)
                mask = mask1.convolve(mask2, output_mask)
                self.assertIsInstance(mask, pygame.mask.Mask, msg)
                self.assertIs(mask, output_mask, msg)
                self.assertEqual(mask.get_size(), output_size, msg)