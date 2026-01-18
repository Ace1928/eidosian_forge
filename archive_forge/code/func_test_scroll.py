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
def test_scroll(self):
    scrolls = [(8, 2, 3), (16, 2, 3), (24, 2, 3), (32, 2, 3), (32, -1, -3), (32, 0, 0), (32, 11, 0), (32, 0, 11), (32, -11, 0), (32, 0, -11), (32, -11, 2), (32, 2, -11)]
    for bitsize, dx, dy in scrolls:
        surf = pygame.Surface((10, 10), 0, bitsize)
        surf.fill((255, 0, 0))
        surf.fill((0, 255, 0), (2, 2, 2, 2))
        comp = surf.copy()
        comp.blit(surf, (dx, dy))
        surf.scroll(dx, dy)
        w, h = surf.get_size()
        for x in range(w):
            for y in range(h):
                with self.subTest(x=x, y=y):
                    self.assertEqual(surf.get_at((x, y)), comp.get_at((x, y)), '%s != %s, bpp:, %i, x: %i, y: %i' % (surf.get_at((x, y)), comp.get_at((x, y)), bitsize, dx, dy))
    surf = pygame.Surface((20, 13), 0, 32)
    surf.fill((255, 0, 0))
    surf.fill((0, 255, 0), (7, 1, 6, 6))
    comp = surf.copy()
    clip = Rect(3, 1, 8, 14)
    surf.set_clip(clip)
    comp.set_clip(clip)
    comp.blit(surf, (clip.x + 2, clip.y + 3), surf.get_clip())
    surf.scroll(2, 3)
    w, h = surf.get_size()
    for x in range(w):
        for y in range(h):
            self.assertEqual(surf.get_at((x, y)), comp.get_at((x, y)))
    spot_color = (0, 255, 0, 128)
    surf = pygame.Surface((4, 4), pygame.SRCALPHA, 32)
    surf.fill((255, 0, 0, 255))
    surf.set_at((1, 1), spot_color)
    surf.scroll(dx=1)
    self.assertEqual(surf.get_at((2, 1)), spot_color)
    surf.scroll(dy=1)
    self.assertEqual(surf.get_at((2, 2)), spot_color)
    surf.scroll(dy=1, dx=1)
    self.assertEqual(surf.get_at((3, 3)), spot_color)
    surf.scroll(dx=-3, dy=-3)
    self.assertEqual(surf.get_at((0, 0)), spot_color)