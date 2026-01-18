import unittest
import pygame
import pygame.gfxdraw
from pygame.locals import *
from pygame.tests.test_utils import SurfaceSubclass
def test_vline(self):
    """vline(surface, x, y1, y2, color): return None"""
    fg = self.foreground_color
    bg = self.background_color
    x = 50
    starty = 10
    stopy = 80
    fg_test_points = [(x, starty), (x, stopy), (x, (stopy - starty) // 2)]
    bg_test_points = [(x, starty - 1), (x, stopy + 1), (x - 1, starty), (x + 1, starty), (x - 1, stopy), (x + 1, stopy)]
    for surf in self.surfaces:
        fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
        bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
        pygame.gfxdraw.vline(surf, x, starty, stopy, fg)
        for posn in fg_test_points:
            self.check_at(surf, posn, fg_adjusted)
        for posn in bg_test_points:
            self.check_at(surf, posn, bg_adjusted)