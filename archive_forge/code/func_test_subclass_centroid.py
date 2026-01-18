from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_centroid(self):
    """Ensures centroid works for subclassed Masks."""
    expected_centroid = (0, 0)
    mask_size = (3, 2)
    mask = SubMask((3, 2))
    centroid = mask.centroid()
    self.assertEqual(centroid, expected_centroid)