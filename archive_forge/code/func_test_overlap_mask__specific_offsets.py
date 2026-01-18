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
def test_overlap_mask__specific_offsets(self):
    """Ensure an offset overlap_mask's mask is correctly calculated.

        Testing the specific case of:
            -both masks are wider than 32 bits
            -a positive offset is used
            -the mask calling overlap_mask() is wider than the mask passed in
        """
    mask1 = pygame.mask.Mask((65, 5), fill=True)
    mask2 = pygame.mask.Mask((33, 3), fill=True)
    expected_mask = pygame.Mask(mask1.get_size())
    rect1 = mask1.get_rect()
    rect2 = mask2.get_rect()
    corner_rect = rect1.inflate(-2, -2)
    for corner in ('topleft', 'topright', 'bottomright', 'bottomleft'):
        setattr(rect2, corner, getattr(corner_rect, corner))
        offset = rect2.topleft
        msg = f'offset={offset}'
        overlap_rect = rect1.clip(rect2)
        expected_mask.clear()
        expected_mask.draw(pygame.Mask(overlap_rect.size, fill=True), overlap_rect.topleft)
        overlap_mask = mask1.overlap_mask(mask2, offset)
        self.assertIsInstance(overlap_mask, pygame.mask.Mask, msg)
        assertMaskEqual(self, overlap_mask, expected_mask, msg)