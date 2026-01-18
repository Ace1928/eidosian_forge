import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_premul_alpha(self):
    Color = pygame.color.Color
    color0 = Color(0, 0, 0, 0)
    alpha0 = Color(255, 255, 255, 0)
    alpha49 = Color(255, 0, 0, 49)
    alpha67 = Color(0, 255, 0, 67)
    alpha73 = Color(0, 0, 255, 73)
    alpha128 = Color(255, 255, 255, 128)
    alpha199 = Color(255, 255, 255, 199)
    alpha255 = Color(128, 128, 128, 255)
    self.assertTrue(isinstance(color0.premul_alpha(), Color))
    self.assertEqual(alpha0.premul_alpha(), Color(0, 0, 0, 0))
    self.assertEqual(alpha49.premul_alpha(), Color(49, 0, 0, 49))
    self.assertEqual(alpha67.premul_alpha(), Color(0, 67, 0, 67))
    self.assertEqual(alpha73.premul_alpha(), Color(0, 0, 73, 73))
    self.assertEqual(alpha128.premul_alpha(), Color(128, 128, 128, 128))
    self.assertEqual(alpha199.premul_alpha(), Color(199, 199, 199, 199))
    self.assertEqual(alpha255.premul_alpha(), Color(128, 128, 128, 255))
    test_colors = [(200, 30, 74), (76, 83, 24), (184, 21, 6), (74, 4, 74), (76, 83, 24), (184, 21, 234), (160, 30, 74), (96, 147, 204), (198, 201, 60), (132, 89, 74), (245, 9, 224), (184, 112, 6)]
    for r, g, b in test_colors:
        for a in range(255):
            with self.subTest(r=r, g=g, b=b, a=a):
                alpha = a / 255.0
                self.assertEqual(Color(r, g, b, a).premul_alpha(), Color((r + 1) * a >> 8, (g + 1) * a >> 8, (b + 1) * a >> 8, a))