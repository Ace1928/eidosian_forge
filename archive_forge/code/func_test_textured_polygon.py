import unittest
import pygame
import pygame.gfxdraw
from pygame.locals import *
from pygame.tests.test_utils import SurfaceSubclass
def test_textured_polygon(self):
    """textured_polygon(surface, points, texture, tx, ty): return None"""
    w, h = self.default_size
    fg = self.foreground_color
    bg = self.background_color
    tx = 0
    ty = 0
    texture = pygame.Surface((w + tx, h + ty), 0, 24)
    texture.fill(fg, (0, 0, w, h))
    points = [(10, 80), (10, 15), (92, 25), (92, 80)]
    fg_test_points = [(points[1][0] + 30, points[1][1] + 40)]
    bg_test_points = [(points[0][0] - 1, points[0][1]), (points[0][0], points[0][1] + 1), (points[0][0] - 1, points[0][1] + 1), (points[3][0] + 1, points[3][1]), (points[3][0], points[3][1] + 1), (points[3][0] + 1, points[3][1] + 1), (points[2][0] + 1, points[2][1]), (points[1][0] - 1, points[1][1]), (points[1][0], points[1][1] - 1), (points[1][0] - 1, points[1][1] - 1)]
    for surf in self.surfaces[1:]:
        fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
        bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
        pygame.gfxdraw.textured_polygon(surf, points, texture, -tx, -ty)
        for posn in fg_test_points:
            self.check_at(surf, posn, fg_adjusted)
        for posn in bg_test_points:
            self.check_at(surf, posn, bg_adjusted)
    texture = pygame.Surface(self.default_size, SRCALPHA, 32)
    self.assertRaises(ValueError, pygame.gfxdraw.textured_polygon, self.surfaces[0], points, texture, 0, 0)