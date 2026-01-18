from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_copy(self):
    """Ensures copy correctly handles zero sized masks."""
    for expected_size in ((11, 0), (0, 11), (0, 0)):
        mask = pygame.mask.Mask(expected_size)
        mask_copy = mask.copy()
        self.assertIsInstance(mask_copy, pygame.mask.Mask)
        self.assertIsNot(mask_copy, mask)
        assertMaskEqual(self, mask_copy, mask)