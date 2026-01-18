import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_set_num_channels(self):
    mixer.init()
    default_num_channels = mixer.get_num_channels()
    for i in range(1, default_num_channels + 1):
        mixer.set_num_channels(i)
        self.assertEqual(mixer.get_num_channels(), i)