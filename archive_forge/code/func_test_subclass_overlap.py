from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_overlap(self):
    """Ensures overlap works for subclassed Masks."""
    expected_pos = (0, 0)
    mask_size = (2, 3)
    masks = (pygame.mask.Mask(fill=True, size=mask_size), SubMask(mask_size, True))
    arg_masks = (pygame.mask.Mask(fill=True, size=mask_size), SubMask(mask_size, True))
    for mask in masks:
        for arg_mask in arg_masks:
            overlap_pos = mask.overlap(arg_mask, (0, 0))
            self.assertEqual(overlap_pos, expected_pos)