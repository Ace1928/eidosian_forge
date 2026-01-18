import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
def test_overlap_check(self):
    bgc = (0, 0, 0, 255)
    rectc_left = (128, 64, 32, 255)
    rectc_right = (255, 255, 255, 255)
    colors = [(255, 255, 255, 255), (128, 64, 32, 255)]
    overlaps = [(0, 0, 1, 0, (50, 0)), (0, 0, 49, 1, (98, 2)), (0, 0, 49, 49, (98, 98)), (49, 0, 0, 1, (0, 2)), (49, 0, 0, 49, (0, 98))]
    surfs = [pygame.Surface((100, 100), SRCALPHA, 32)]
    surf = pygame.Surface((100, 100), 0, 32)
    surf.set_alpha(255)
    surfs.append(surf)
    surf = pygame.Surface((100, 100), 0, 32)
    surf.set_colorkey((0, 1, 0))
    surfs.append(surf)
    for surf in surfs:
        for s_x, s_y, d_x, d_y, test_posn in overlaps:
            surf.fill(bgc)
            surf.fill(rectc_right, (25, 0, 25, 50))
            surf.fill(rectc_left, (0, 0, 25, 50))
            surf.blit(surf, (d_x, d_y), (s_x, s_y, 50, 50))
            self.assertEqual(surf.get_at(test_posn), rectc_right)