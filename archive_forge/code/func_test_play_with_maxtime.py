import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_play_with_maxtime(self):
    """Test playing a sound with maxtime."""
    channel = self.sound.play(maxtime=200)
    self.assertIsInstance(channel, pygame.mixer.Channel)
    self.assertTrue(channel.get_busy())
    pygame.time.wait(200 + 50)
    self.assertFalse(channel.get_busy())