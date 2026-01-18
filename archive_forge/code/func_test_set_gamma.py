import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
@unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'Needs a not dummy videodriver')
def test_set_gamma(self):
    pygame.display.set_mode((1, 1))
    gammas = [0.25, 0.5, 0.88, 1.0]
    for gamma in gammas:
        with self.subTest(gamma=gamma):
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(pygame.display.set_gamma(gamma), True)
            self.assertEqual(pygame.display.set_gamma(gamma), True)