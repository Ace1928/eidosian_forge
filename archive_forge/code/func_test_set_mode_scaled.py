import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_set_mode_scaled(self):
    surf = pygame.display.set_mode(size=(1, 1), flags=pygame.SCALED, depth=0, display=0)
    winsize = pygame.display.get_window_size()
    self.assertEqual(winsize[0] % surf.get_size()[0], 0, 'window width should be a multiple of the surface width')
    self.assertEqual(winsize[1] % surf.get_size()[1], 0, 'window height should be a multiple of the surface height')
    self.assertEqual(winsize[0] / surf.get_size()[0], winsize[1] / surf.get_size()[1])