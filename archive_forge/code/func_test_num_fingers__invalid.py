import unittest
import os
import pygame
from pygame._sdl2 import touch
from pygame.tests.test_utils import question
def test_num_fingers__invalid(self):
    self.assertRaises(TypeError, touch.get_num_fingers, 'test')
    self.assertRaises(pygame.error, touch.get_num_fingers, -1234)