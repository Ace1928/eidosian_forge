import unittest
import os
import pygame
from pygame._sdl2 import touch
from pygame.tests.test_utils import question
@unittest.skipIf(not has_touchdevice, 'no touch devices found')
def test_num_fingers(self):
    touch.get_num_fingers(touch.get_device(0))