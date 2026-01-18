import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_pause_unpause__before_init(self):
    """
        Ensure exception for Channel.pause() with non-init mixer.
        """
    sound = mixer.Sound(example_path('data/house_lo.wav'))
    channel = sound.play()
    mixer.quit()
    with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
        channel.pause()
    with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
        channel.unpause()