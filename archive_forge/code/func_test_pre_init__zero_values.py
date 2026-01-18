import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_pre_init__zero_values(self):
    mixer.pre_init(22050, -8, 1)
    mixer.pre_init(0, 0, 0)
    mixer.init(allowedchanges=0)
    self.assertEqual(mixer.get_init()[0], 44100)
    self.assertEqual(mixer.get_init()[1], -16)
    self.assertGreaterEqual(mixer.get_init()[2], 2)