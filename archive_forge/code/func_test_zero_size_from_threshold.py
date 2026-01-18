from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_size_from_threshold(self):
    a = [16, 24, 32]
    sizes = ((100, 0), (0, 100), (0, 0))
    for size in sizes:
        for i in a:
            surf = pygame.surface.Surface(size, 0, i)
            surf.fill((100, 50, 200), (20, 20, 20, 20))
            mask = pygame.mask.from_threshold(surf, (100, 50, 200, 255), (10, 10, 10, 255))
            self.assertEqual(mask.count(), 0)
            rects = mask.get_bounding_rects()
            self.assertEqual(rects, [])
        for i in a:
            surf = pygame.surface.Surface(size, 0, i)
            surf2 = pygame.surface.Surface(size, 0, i)
            surf.fill((100, 100, 100))
            surf2.fill((150, 150, 150))
            surf2.fill((100, 100, 100), (40, 40, 10, 10))
            mask = pygame.mask.from_threshold(surf, (0, 0, 0, 0), (10, 10, 10, 255), surf2)
            self.assertIsInstance(mask, pygame.mask.Mask)
            self.assertEqual(mask.count(), 0)
            rects = mask.get_bounding_rects()
            self.assertEqual(rects, [])