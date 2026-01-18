from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_erase(self):
    """Ensures erase works for subclassed Masks."""
    mask_size = (3, 4)
    expected_count = 0
    masks = (pygame.mask.Mask(mask_size, True), SubMask(mask_size, True))
    arg_masks = (pygame.mask.Mask(mask_size, True), SubMask(mask_size, True))
    for mask in masks:
        for arg_mask in arg_masks:
            mask.fill()
            mask.erase(arg_mask, (0, 0))
            self.assertEqual(mask.count(), expected_count)