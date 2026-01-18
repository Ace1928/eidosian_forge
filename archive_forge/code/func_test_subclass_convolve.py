from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_convolve(self):
    """Ensures convolve works for subclassed Masks."""
    width, height = (7, 5)
    mask_size = (width, height)
    expected_count = 0
    expected_size = (max(0, width * 2 - 1), max(0, height * 2 - 1))
    arg_masks = (pygame.mask.Mask(mask_size), SubMask(mask_size))
    output_masks = (pygame.mask.Mask(mask_size), SubMask(mask_size))
    for mask in (pygame.mask.Mask(mask_size), SubMask(mask_size)):
        for arg_mask in arg_masks:
            convolve_mask = mask.convolve(arg_mask)
            self.assertIsInstance(convolve_mask, pygame.mask.Mask)
            self.assertNotIsInstance(convolve_mask, SubMask)
            self.assertEqual(convolve_mask.count(), expected_count)
            self.assertEqual(convolve_mask.get_size(), expected_size)
            for output_mask in output_masks:
                convolve_mask = mask.convolve(arg_mask, output_mask)
                self.assertIsInstance(convolve_mask, pygame.mask.Mask)
                self.assertEqual(convolve_mask.count(), expected_count)
                self.assertEqual(convolve_mask.get_size(), mask_size)
                if isinstance(output_mask, SubMask):
                    self.assertIsInstance(convolve_mask, SubMask)
                else:
                    self.assertNotIsInstance(convolve_mask, SubMask)