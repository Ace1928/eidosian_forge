import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
@unittest.skipIf(True, 'set_palette() not supported in SDL2')
def test_set_palette(self):
    with self.assertRaises(pygame.error):
        palette = [1, 2, 3]
        pygame.display.set_palette(palette)
    pygame.display.set_mode((1024, 768), 0, 8)
    palette = []
    self.assertIsNone(pygame.display.set_palette(palette))
    with self.assertRaises(ValueError):
        palette = 12
        pygame.display.set_palette(palette)
    with self.assertRaises(TypeError):
        palette = [[1, 2], [1, 2]]
        pygame.display.set_palette(palette)
    with self.assertRaises(TypeError):
        palette = [[0, 0, 0, 0, 0]] + [[x, x, x, x, x] for x in range(1, 255)]
        pygame.display.set_palette(palette)
    with self.assertRaises(TypeError):
        palette = 'qwerty'
        pygame.display.set_palette(palette)
    with self.assertRaises(TypeError):
        palette = [[123, 123, 123] * 10000]
        pygame.display.set_palette(palette)
    with self.assertRaises(TypeError):
        palette = [1, 2, 3]
        pygame.display.set_palette(palette)