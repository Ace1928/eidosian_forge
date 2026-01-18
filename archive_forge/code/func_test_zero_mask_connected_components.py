from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_connected_components(self):
    """Ensures connected_components correctly handles zero sized masks."""
    expected_cc_masks = []
    for size in ((11, 0), (0, 10), (0, 0)):
        mask = pygame.mask.Mask(size)
        cc_masks = mask.connected_components()
        self.assertListEqual(cc_masks, expected_cc_masks, f'size={size}')