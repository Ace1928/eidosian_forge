import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
class TestSoundFadeout(unittest.TestCase):

    def setUp(self):
        if mixer.get_init() is None:
            pygame.mixer.init()

    def tearDown(self):
        pygame.mixer.quit()

    def test_fadeout_with_valid_time(self):
        """Tests if fadeout stops sound playback after fading it out over the time argument in milliseconds."""
        filename = example_path(os.path.join('data', 'punch.wav'))
        sound = mixer.Sound(file=filename)
        channel = sound.play()
        channel.fadeout(1000)
        pygame.time.wait(2000)
        self.assertFalse(channel.get_busy())