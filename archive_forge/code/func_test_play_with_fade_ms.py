import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_play_with_fade_ms(self):
    """Test playing a sound with fade_ms."""
    channel = self.sound.play(fade_ms=500)
    self.assertIsInstance(channel, pygame.mixer.Channel)
    self.assertTrue(channel.get_busy())
    pygame.time.wait(250)
    self.assertGreater(channel.get_volume(), 0.3)
    self.assertLess(channel.get_volume(), 0.8)
    pygame.time.wait(300)
    self.assertEqual(channel.get_volume(), 1.0)