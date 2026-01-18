import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
@unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'Needs a not dummy videodriver')
def test_set_gamma__tuple(self):
    pygame.display.set_mode((1, 1))
    gammas = [(0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (0.25, 0.33, 0.44)]
    for r, g, b in gammas:
        with self.subTest(r=r, g=g, b=b):
            self.assertEqual(pygame.display.set_gamma(r, g, b), True)