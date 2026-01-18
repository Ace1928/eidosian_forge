import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_play_once(self):
    """Test playing a sound once."""
    channel = self.sound.play()
    self.assertIsInstance(channel, pygame.mixer.Channel)
    self.assertTrue(channel.get_busy())